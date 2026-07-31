"""End-to-end plumbing: corpus (synthetic, mirrors real schema) -> window -> noise -> collate ->
model -> loss, on CPU with a tiny random-init model. Mirrors tests/test_insc_smoke.py. Proves
the whole pipeline survives shape/index mismatches before spending any SLURM time on it."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.char_bert import CharBertConfig, CharBertEncoder
from train.loss import compute_loss
from trimeter.collate import pack_windows
from trimeter.data.corpus import Line, Stretch
from trimeter.data.window import sample_window
from trimeter.noise import TrimeterNoiseConfig


def _fake_line(n, length, seed):
    rng = np.random.default_rng(seed)
    chars = rng.integers(0, 24, length).astype(np.uint8)
    boundary = np.zeros(length, np.uint8)
    boundary[-1] = 2
    if length > 1:
        boundary[: length - 1][rng.random(length - 1) < 0.3] = 1
    return Line(str(n), "x" * length, chars, boundary,
                np.zeros(length, np.uint8), np.zeros(length, np.uint8), np.zeros(length, np.uint8))


def _fake_stretch(n_lines, urn, seed0=0):
    rng = np.random.default_rng(seed0)
    lengths = rng.integers(15, 55, n_lines)
    return Stretch(urn, "Testus", "Πεῖρα", [_fake_line(i, int(L), seed0 * 100 + i)
                                             for i, L in enumerate(lengths)])


def test_full_pipeline_forward_and_loss():
    torch.manual_seed(0)
    stretches = [_fake_stretch(12, "tlg9999.tlg001", 0),
                 _fake_stretch(8, "tlg9999.tlg002", 1),
                 _fake_stretch(20, "tlg9999.tlg003", 2)]

    rng = np.random.default_rng(0)
    T = 512
    windows = []
    for _ in range(4):
        s = stretches[int(rng.integers(len(stretches)))]
        w = sample_window(s, rng, w_min=1, w_max=10, T_char=T, safety_margin=64)
        assert w is not None
        windows.append(w)

    ncfg = TrimeterNoiseConfig(recon_width=20)
    batch, metas = pack_windows(windows, ncfg, T, rng)
    assert len(metas) == 4
    for k in ("input_ids", "boundary", "dia", "punct", "seg_id", "labels", "loss_w",
              "bnd_lab", "dia_lab", "cap_lab", "punct_lab"):
        assert batch[k].shape == (4, T), k

    cfg = CharBertConfig(attn_impl="sdpa", d_model=32, n_heads=4, depth=2, char_window=0)
    model = CharBertEncoder(cfg)
    out = model(batch)
    assert out["char"].shape == (4, T, cfg.n_char_ids)

    loss, logs = compute_loss(out, batch, lam=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert "char" in logs and "total" in logs
    # aux heads (bnd/dia/cap/punct) now get real supervision from the synthetic-damage ground
    # truth, not just the always-empty labels of the original (letters-only) design
    assert "bnd" in logs and "dia" in logs and "cap" in logs and "punct" in logs
