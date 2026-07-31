"""Dev-stall-driven finetune (v2): same data/noise recipe as finetune.py, but the
schedule is decided by the PHI digit-4 dev set instead of a fixed step budget:

  warmup -> constant peak LR until dev masked-bits/char stalls (window/eps as in
  GreekCharBERT pretraining) -> cosine decay over decay_len steps, with dev
  early-stop (patience evals without improvement) -> best.pt is the model of record.

Dev signal: masked bits/char (eval.intrinsic.evaluate) on a fixed sample of digit-4
val segments, minus the pretraining-contamination exclusion list, every eval_every
steps. Test (digit 3) is never touched here.

  torchrun --nproc_per_node=4 insc_train/finetune_v2.py --config configs/finetune_fold0_v2.json
"""
from __future__ import annotations

import argparse, contextlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from model.char_bert import CharBertConfig, CharBertEncoder, num_params
from meta_vocab import N_REGION, N_CENTURY
from train.loss import compute_loss
from train.noising import NoiseConfig
from train.data import DataConfig, TierSpec
from train.train import ddp_setup, save_ckpt

sys.path.insert(2, str(Path(__file__).resolve().parent))
from finetune import make_loader, lr_mult  # noqa: F401  (reuse loader machinery)


def get_loader(domain):
    """Domain-select the segment loader (same record interface). 'both' chains
    iphi + papyri segments -- both loaders emit the same dict keys (phi_id/seg/
    split/region/tpq/taq + planes), so downstream code (dev_records, exclude
    filtering) is domain-agnostic."""
    if domain == "papyri":
        from papyri import load
    elif domain == "both":
        from iphi import load as iphi_load
        from papyri import load as pap_load

        def load(split=None, min_len=32, max_records=None):
            return (iphi_load(split=split, min_len=min_len, max_records=max_records)
                    + pap_load(split=split, min_len=min_len, max_records=max_records))
    else:
        from iphi import load
    return load

PLANES = ("chars", "boundary", "dia", "cap", "punct")


def lr_mult_dyn(step, warmup, anneal_start, decay_len):
    """Warmup -> flat peak until anneal_start -> cosine to 0.02 over decay_len."""
    if step < warmup:
        return step / max(warmup, 1)
    if anneal_start is None or step < anneal_start:
        return 1.0
    t = min((step - anneal_start) / max(decay_len, 1), 1.0)
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))


def stalled(bpc, window=8, eps=0.002):
    if len(bpc) < window:
        return False
    half = window // 2
    return min(bpc[-half:]) > min(bpc[-window:-half]) - eps


def dev_records(exclude_path, n=256, seed=1234, loader=None):
    """Fixed sample of clean digit-4 val segments as eval records."""
    excl = set()
    if exclude_path:
        j = json.loads(Path(exclude_path).read_text())
        excl = {(str(x[0]), int(x[1])) for x in j["contaminated"]}
    recs = [r for r in loader(split="val", min_len=50)
            if 64 <= len(r["chars"]) <= 1500
            and (str(r["phi_id"]), int(r["seg"])) not in excl]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
    return [{p: np.asarray(recs[i][p]) for p in PLANES} for i in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])
    # If the config pins a test/val digit (rotating-fold restoration configs), it is
    # authoritative -- overrides whatever INSC_TEST_DIGIT/INSC_VAL_DIGIT (if any) the
    # launcher exported, so running this file directly (not via insc_finetune_fold.sbatch)
    # can't silently fall back to the default 3/4 split.
    if "test_digit" in cfg:
        os.environ["INSC_TEST_DIGIT"] = str(cfg["test_digit"])
    if "val_digit" in cfg:
        os.environ["INSC_VAL_DIGIT"] = str(cfg["val_digit"])
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")

    T = cfg["seq_len"]; rows = cfg["micro_batch"]; accum = cfg.get("grad_accum", 1)
    hard_max = cfg.get("total_steps", 12000)          # safety cap, not the budget
    decay_len = cfg.get("decay_len", 1500)
    warmup = cfg.get("warmup", 200)
    eval_every = cfg.get("eval_every", 250)
    patience = cfg.get("anneal_patience", 3)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    meta_condition = cfg.get("meta_condition", False)
    mcfg = CharBertConfig(attn_impl=cfg.get("attn", "flex"), d_model=cfg["d_model"],
                          n_heads=cfg["d_model"] // 64, depth=cfg["depth"],
                          char_window=cfg["char_window"], qk_norm=True,
                          n_region=N_REGION if meta_condition else 0,
                          n_century=N_CENTURY if meta_condition else 0)
    model = CharBertEncoder(mcfg).to(device)

    step0 = 0
    ckpt = out / "last.pt"
    sd = None
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"]); step0 = sd["step"]
        if rank == 0:
            print(f"resumed finetune from step {step0}")
    elif cfg["torso"] == "random":
        if rank == 0:
            print("RANDOM INIT (ablation): no pretrained torso loaded")
    else:
        torso = os.path.expandvars(cfg["torso"])
        tsd = torch.load(torso, map_location=device)
        # torso is always a non-conditioned pretraining checkpoint (no e_region/e_century
        # weights); strict=False when meta_condition is on so those two new tables stay at
        # their random init instead of failing the load. Any other missing/unexpected key
        # would still be a real bug -- check explicitly rather than blanket-swallowing.
        res = model.load_state_dict(tsd["model"], strict=not meta_condition)
        if meta_condition:
            assert set(res.missing_keys) <= {"e_region.weight", "e_century.weight"}, res.missing_keys
            assert not res.unexpected_keys, res.unexpected_keys
        if rank == 0:
            print(f"initialized from torso {torso} (pretrain step {tsd.get('step')})"
                  + (" [meta_condition: region/century embeddings randomly init'd]"
                     if meta_condition else ""))
    if rank == 0:
        print(f"params={num_params(model)/1e6:.1f}M T={T} rows={rows}x{accum} world={world} "
              f"hard_max={hard_max} decay_len={decay_len}")
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model

    decay_p = [p for p in model.parameters() if p.ndim >= 2]
    nodecay_p = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": cfg.get("wd", 0.1)},
                             {"params": nodecay_p, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if sd is not None and "opt" in sd:
        try:
            opt.load_state_dict(sd["opt"])
        except Exception:
            pass

    nc = cfg.get("noise", {})
    ncfg = NoiseConfig(w_span=nc.get("w_span", 0.45), w_word=nc.get("w_word", 0.15),
                       w_elastic=nc.get("w_elastic", 0.0), w_iid=nc.get("w_iid", 0.15),
                       w_halfword=nc.get("w_halfword", 0.15),
                       w_substitute=nc.get("w_substitute", 0.10),
                       span_mean=nc.get("span_mean", 4.0), span_max=nc.get("span_max", 12))

    ins = os.path.expandvars(cfg["iphi_shards"])
    gcb = os.path.expandvars(cfg["gcb_shards"])
    mix = cfg.get("mix", {"iphi": 1.0, "iphi_syn": 0.5, "gold": 0.15, "silver": 0.15})
    tiers = {
        "iphi":     TierSpec(ins, mix.get("iphi", 1.0), tier_filter="iphi"),
        "iphi_syn": TierSpec(ins, mix.get("iphi_syn", 0.5), tier_filter="iphi_syn"),
        "gold":     TierSpec(gcb, mix.get("gold", 0.15), tier_filter="pristine"),
        "silver":   TierSpec(gcb, mix.get("silver", 0.15), tier_filter="repaired"),
    }
    if cfg.get("pap_shards"):
        # second in-domain corpus mixed alongside iphi_shards (combined PHI+TM run) --
        # pap_punct shards use the same generic tier_filter="iphi" label internally
        pap = os.path.expandvars(cfg["pap_shards"])
        tiers["papyri"] = TierSpec(pap, mix.get("papyri", 1.0), tier_filter="iphi")
    dcfg = DataConfig(tiers=tiers, window_chars=T, seed=cfg["seed"] + step0)
    it = iter(make_loader(dcfg, ncfg, T, rows, cfg["seed"] + step0, rank, world,
                          cfg.get("num_workers", 8)))

    # dev eval fixture (rank 0 only; fixed sample -> comparable series). For the
    # combined domain, the merged fixture drives the stall/anneal/early-stop decision;
    # separate per-domain fixtures are ALSO tracked purely for reporting (so a joint
    # run's transfer-learning effect on each domain is visible, not just the average).
    dev_domain = cfg.get("dev_domain", "iphi")
    recs = recs_iphi = recs_pap = None
    if rank == 0:
        excl = os.path.expandvars(cfg.get("dev_exclude", "")) or None
        recs = dev_records(excl, n=cfg.get("eval_n", 256), loader=get_loader(dev_domain))
        print(f"dev eval fixture ({dev_domain}): {len(recs)} clean digit-4 segments")
        if dev_domain == "both":
            recs_iphi = dev_records(excl, n=cfg.get("eval_n", 256), loader=get_loader("iphi"))
            recs_pap = dev_records(excl, n=cfg.get("eval_n", 256), loader=get_loader("papyri"))
            print(f"  + per-domain report fixtures: {len(recs_iphi)} iphi, {len(recs_pap)} papyri")
    from eval.intrinsic import evaluate

    marker = out / "anneal_start.json"
    anneal_start = None
    if marker.exists():
        anneal_start = json.loads(marker.read_text())["step"]
        if rank == 0:
            print(f"anneal marker: from step {anneal_start}")

    model.train()
    t0 = time.time(); seen = 0
    metrics_f = out / "metrics.jsonl"
    dev_bpc = []
    stop = False
    for step in range(step0, hard_max):
        if anneal_start is not None and step >= anneal_start + decay_len:
            break
        lr = cfg["lr"] * lr_mult_dyn(step, warmup, anneal_start, decay_len)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(it)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            sync = (model.no_sync() if (is_ddp and micro < accum - 1)
                    else contextlib.nullcontext())
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out_h = fwd(batch)
                    loss, logs = compute_loss(out_h, batch, lam=cfg.get("lam", 0.1))
                (loss / accum).backward()
            seen += int((batch["seg_id"] > 0).sum()) * world
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            opt.step()

        if rank == 0 and step % cfg.get("log_every", 20) == 0:
            rec = dict(step=step, lr=round(lr, 7), gnorm=round(float(gnorm), 3),
                       chps=round(seen / (time.time() - t0) / 1e6, 3),
                       anneal=anneal_start is not None, **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if step > step0 and step % eval_every == 0:
            if rank == 0:
                core = model.module if is_ddp else model
                save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(),
                               step=step + 1, cfg=cfg), ckpt)
                m = evaluate(core, recs, device)
                m["step"] = step; m["split"] = "dev4"
                if recs_iphi is not None:
                    m["bpc_iphi"] = evaluate(core, recs_iphi, device)["bits_per_char"]
                    m["bpc_papyri"] = evaluate(core, recs_pap, device)["bits_per_char"]
                print("  DEV " + json.dumps(m), flush=True)
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
                dev_bpc.append((step, m["bits_per_char"]))
                series = [b for _, b in dev_bpc]
                if m["bits_per_char"] <= min(series):
                    import shutil
                    shutil.copyfile(ckpt, out / "best.pt")
                    print(f"  new best.pt (dev bpc {m['bits_per_char']})", flush=True)
                if anneal_start is None and step >= warmup + eval_every * cfg.get("stall_window", 8) \
                        and stalled(series, cfg.get("stall_window", 8), cfg.get("stall_eps", 0.002)):
                    anneal_start = step + 1
                    marker.write_text(json.dumps(dict(step=anneal_start, reason="dev-stall")))
                    print(f"  DEV-STALL: annealing from {anneal_start}, ends "
                          f"{anneal_start + decay_len}", flush=True)
                if anneal_start is not None and patience > 0 \
                        and step >= anneal_start + decay_len // 2:
                    ann = [(s, b) for s, b in dev_bpc if s > anneal_start]
                    if len(ann) > patience:
                        bi = min(range(len(ann)), key=lambda i: ann[i][1])
                        if len(ann) - 1 - bi >= patience:
                            stop = True
                            print(f"  DEV EARLY-STOP (best at step {ann[bi][0]}); best.pt "
                                  f"is the model of record", flush=True)
            if is_ddp:
                import torch.distributed as dist
                t_a = torch.tensor([-1 if anneal_start is None else anneal_start,
                                    1 if stop else 0], dtype=torch.long, device=device)
                dist.broadcast(t_a, 0)
                v = int(t_a[0].item())
                anneal_start = None if v < 0 else v
                stop = bool(int(t_a[1].item()))
            if stop:
                break
        model.train()

    if rank == 0:
        core = model.module if is_ddp else model
        end = step + 1
        state = dict(model=core.state_dict(), opt=opt.state_dict(), step=end, cfg=cfg,
                     stopped_early=stop)
        save_ckpt(state, out / "final.pt")
        save_ckpt(state, ckpt)
        print(f"FINETUNE DONE (end step {end}" + (", EARLY-STOPPED — use best.pt" if stop else "") + ")")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
