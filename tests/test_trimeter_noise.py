"""Damage sampler correctness: span-length bounds, anchor mixture, elastic-rebuild label
semantics, gap-channel forcing, and the vertical-tear vs independent correlation property that
distinguishes the two multi-line damage modes."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trimeter.data.corpus import Line, Stretch
from trimeter.data.window import build_window
from trimeter.noise import (BLANK_ID, MASK_ID, UNK_BND, UNK_DIA, UNK_PUNCT,
                            TrimeterNoiseConfig, bucket_key, rebuild_window, sample_damage,
                            sample_span, sample_span_both)


def _fake_line(n, length, seed):
    rng = np.random.default_rng(seed)
    chars = rng.integers(0, 24, length).astype(np.uint8)
    boundary = np.zeros(length, np.uint8)
    boundary[-1] = 2
    return Line(str(n), "x" * length, chars, boundary,
                np.zeros(length, np.uint8), np.zeros(length, np.uint8), np.zeros(length, np.uint8))


def _fake_stretch(lengths, urn="tlg9999.tlg001"):
    return Stretch(urn, "Testus", "Πεῖρα", [_fake_line(i, L, i) for i, L in enumerate(lengths)])


def test_span_length_never_exceeds_recon_width_or_line_len():
    cfg = TrimeterNoiseConfig(recon_width=40)
    rng = np.random.default_rng(0)
    for line_len in range(1, 80):
        for _ in range(10):
            s, e, a = sample_span(line_len, rng, cfg)
            L = e - s
            assert 0 < L <= min(cfg.recon_width, line_len)
            assert 0 <= s < e <= line_len


def test_anchor_mixture_hits_all_three():
    cfg = TrimeterNoiseConfig()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(500):
        _, _, a = sample_span(40, rng, cfg)
        seen.add(a)
    assert seen == {"begin", "end", "mid"}


def test_rebuild_labels_match_elastic_semantics():
    cfg = TrimeterNoiseConfig(recon_width=40, p_clean_context=1.0, p_independent=0.0,
                              p_vertical_tear=0.0)
    stretch = _fake_stretch([20, 30, 15])
    win = build_window(stretch, 0, 3, 1)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    assert mode == "clean"
    assert len(spans) == 1
    _, s, e, is_p, _ = spans[0]
    L = e - s
    out = rebuild_window(win, spans, rng, cfg)
    meta = out["span_meta"][0]
    M = cfg.recon_width
    start = meta["start"]
    assert out["input_ids"][start:start + M].eq(MASK_ID).all()
    assert (out["labels"][start:start + L] == win.chars[win.offsets[1][0] + s: win.offsets[1][0] + e]).all()
    assert (out["labels"][start + L:start + M] == BLANK_ID).all()
    # non-gap positions unchanged
    before = out["input_ids"][:start]
    orig_before = win.chars[: win.offsets[1][0] + s]
    assert (before.numpy() == orig_before).all()


def test_aux_labels_supervise_real_ground_truth_inside_gap():
    """Unlike PHI restoration (true gap content genuinely unknown), this corpus IS fully known
    clean text -- boundary/dia/punct/cap aux labels inside the gap must hold the REAL
    underlying value for the true [0,L) content, -100 for the [L,M) padding tail (no real
    position exists there)."""
    cfg = TrimeterNoiseConfig(recon_width=30, p_clean_context=1.0, p_independent=0.0,
                              p_vertical_tear=0.0, p_bnd_full=0.0, p_bnd_none=0.0,
                              p_dia_full=0.0, p_dia_none=0.0, p_punct_full=0.0, p_punct_none=0.0)
    stretch = _fake_stretch([20, 35, 15])
    win = build_window(stretch, 0, 3, 1)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    _, s, e, is_p, _ = spans[0]
    L = e - s
    line_s = win.offsets[1][0]
    out = rebuild_window(win, spans, rng, cfg)
    meta = out["span_meta"][0]
    start, M = meta["start"], meta["M"]

    true_bnd = win.boundary[line_s + s: line_s + e]
    true_dia = win.dia[line_s + s: line_s + e]
    true_punct = win.punct[line_s + s: line_s + e]
    true_cap = win.cap[line_s + s: line_s + e]
    assert (out["bnd_lab"][start:start + L].numpy() == true_bnd).all()
    assert (out["dia_lab"][start:start + L].numpy() == true_dia).all()
    assert (out["punct_lab"][start:start + L].numpy() == true_punct).all()
    assert (out["cap_lab"][start:start + L].numpy() == true_cap).all()
    # padding tail (no real position) is never supervised
    assert (out["bnd_lab"][start + L:start + M] == -100).all()
    assert (out["dia_lab"][start + L:start + M] == -100).all()
    assert (out["punct_lab"][start + L:start + M] == -100).all()
    assert (out["cap_lab"][start + L:start + M] == -100).all()


def test_aux_labels_not_supervised_when_channel_visible_outside_gap():
    """Outside the gap, a channel that's fully KEPT as input must not also be an aux-loss
    target -- copying through a visible value teaches the model nothing (same convention as
    train/collate.py)."""
    cfg = TrimeterNoiseConfig(recon_width=15, p_clean_context=1.0, p_independent=0.0,
                              p_vertical_tear=0.0, p_bnd_full=1.0, p_bnd_none=0.0,
                              p_dia_full=1.0, p_dia_none=0.0, p_punct_full=1.0, p_punct_none=0.0)
    stretch = _fake_stretch([25, 30])
    win = build_window(stretch, 0, 2, 0)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    out = rebuild_window(win, spans, rng, cfg)
    meta = out["span_meta"][0]
    start, M = meta["start"], meta["M"]
    outside = torch.cat([out["bnd_lab"][:start], out["bnd_lab"][start + M:]])
    assert (outside == -100).all(), "fully-visible boundary channel outside the gap must not be supervised"


def test_cap_always_supervised_inside_gap_never_outside():
    """cap has no input channel at all (always UNK as input), so it's a prediction target
    wherever real content exists inside a gap, regardless of any channel-keep mixture -- and
    never supervised outside a gap (nothing there is ever masked)."""
    cfg = TrimeterNoiseConfig(recon_width=20, p_clean_context=1.0, p_independent=0.0,
                              p_vertical_tear=0.0)
    stretch = _fake_stretch([22, 28])
    win = build_window(stretch, 0, 2, 0)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    _, s, e, _, _ = spans[0]
    L = e - s
    out = rebuild_window(win, spans, rng, cfg)
    meta = out["span_meta"][0]
    start, M = meta["start"], meta["M"]
    assert (out["cap_lab"][start:start + L] != -100).all()
    outside = torch.cat([out["cap_lab"][:start], out["cap_lab"][start + M:]])
    assert (outside == -100).all()


def test_gap_boundary_dia_punct_forced_unknown():
    cfg = TrimeterNoiseConfig(p_clean_context=1.0, p_independent=0.0, p_vertical_tear=0.0,
                              p_bnd_full=1.0, p_bnd_none=0.0, p_dia_full=1.0, p_dia_none=0.0,
                              p_punct_full=1.0, p_punct_none=0.0)
    stretch = _fake_stretch([25, 30])
    win = build_window(stretch, 0, 2, 0)
    rng = np.random.default_rng(0)
    spans, _ = sample_damage(win, rng, cfg)
    out = rebuild_window(win, spans, rng, cfg)
    meta = out["span_meta"][0]
    start, M = meta["start"], meta["M"]
    assert (out["boundary"][start:start + M] == UNK_BND).all()
    assert (out["dia"][start:start + M] == UNK_DIA).all()
    assert (out["punct"][start:start + M] == UNK_PUNCT).all()


def test_clean_context_mode_touches_only_primary():
    cfg = TrimeterNoiseConfig(p_clean_context=1.0, p_independent=0.0, p_vertical_tear=0.0)
    stretch = _fake_stretch([20] * 8)
    win = build_window(stretch, 0, 8, 3)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    assert mode == "clean"
    assert len(spans) == 1 and spans[0][0] == 3


def test_independent_mode_is_uncorrelated():
    cfg = TrimeterNoiseConfig(p_clean_context=0.0, p_independent=1.0, p_vertical_tear=0.0,
                              p_secondary_line=1.0)
    # varying lengths so independent spans have room to disagree
    stretch = _fake_stretch([15, 40, 20, 55, 25, 45, 30, 60])
    win = build_window(stretch, 0, 8, 3)
    starts = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        spans, mode = sample_damage(win, rng, cfg)
        assert mode == "independent"
        for i, s, e, is_p, a in spans:
            if not is_p:
                starts.append(win.offsets[i][0] + s)
    assert len(starts) > 10
    assert np.std(starts) > 5, "independent damage should not be aligned across lines"


def test_vertical_tear_produces_correlated_contiguous_run():
    cfg = TrimeterNoiseConfig(p_clean_context=0.0, p_independent=0.0, p_vertical_tear=1.0,
                              tear_offset_jitter=0, tear_width_jitter_frac=0.0,
                              tear_k_min=4, tear_k_max=4)
    stretch = _fake_stretch([35] * 8)  # equal lengths -> no line "misses" the tear
    win = build_window(stretch, 0, 8, 3)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    assert mode == "vertical_tear"
    assert len(spans) > 1, "tear should damage more than just the primary line"
    idxs = sorted(i for i, *_ in spans)
    assert idxs == list(range(idxs[0], idxs[-1] + 1)), "damaged lines must be contiguous"
    # relative-to-line-start offsets are the actual alignment property (absolute offsets in the
    # concatenated window necessarily grow line-by-line regardless of alignment, so only a
    # per-line-relative comparison is meaningful here)
    rel_starts = [s for i, s, e, is_p, a in spans]
    assert np.std(rel_starts) < 3.0


def test_short_line_missed_by_tear():
    cfg = TrimeterNoiseConfig(p_clean_context=0.0, p_independent=0.0, p_vertical_tear=1.0,
                              tear_offset_jitter=0, tear_width_jitter_frac=0.0,
                              tear_k_min=2, tear_k_max=2, anchor_begin_p=0.0, anchor_end_p=1.0,
                              anchor_mid_p=0.0)
    # primary line long (damage anchored at end -> high column), secondary line very short
    stretch = _fake_stretch([50, 3])
    win = build_window(stretch, 0, 2, 0)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    idxs = [i for i, *_ in spans]
    assert 1 not in idxs, "short secondary line should be missed by a high-column tear"


def test_sample_span_both_leaves_visible_middle_and_no_overlap():
    cfg = TrimeterNoiseConfig(min_middle_letters=2, recon_width=60)
    rng = np.random.default_rng(0)
    for line_len in (10, 20, 37, 51, 65):
        for _ in range(50):
            spans = sample_span_both(line_len, rng, cfg)
            if len(spans) == 1:
                continue  # degenerate too-short-line fallback
            (s1, e1, a1), (s2, e2, a2) = spans
            assert a1 == "both_begin" and a2 == "both_end"
            assert s1 == 0 and e2 == line_len
            assert e1 <= s2, "begin/end gaps must not overlap"
            assert s2 - e1 >= cfg.min_middle_letters, "middle strip must stay visible"


def test_sample_span_both_degenerate_short_line():
    cfg = TrimeterNoiseConfig(min_middle_letters=2)
    rng = np.random.default_rng(0)
    spans = sample_span_both(2, rng, cfg)  # too short to leave a visible middle
    assert len(spans) == 1


def test_both_anchor_produces_two_primary_spans_for_the_line():
    cfg = TrimeterNoiseConfig(anchor_begin_p=0.0, anchor_end_p=0.0, anchor_mid_p=0.0,
                              p_clean_context=1.0, p_independent=0.0, p_vertical_tear=0.0,
                              min_middle_letters=2)
    stretch = _fake_stretch([40, 40, 40])
    win = build_window(stretch, 0, 3, 1)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    primaries = [sp for sp in spans if sp[3]]
    assert len(primaries) == 2
    assert {sp[4] for sp in primaries} == {"both_begin", "both_end"}
    assert bucket_key(spans) == "both"


def test_bucket_key_single_anchor_cases():
    assert bucket_key([(0, 0, 5, True, "begin")]) == "begin"
    assert bucket_key([(0, 0, 5, True, "end")]) == "end"
    assert bucket_key([(0, 0, 5, True, "mid")]) == "mid"
    assert bucket_key([(0, 0, 5, True, "both_begin"), (0, 30, 35, True, "both_end")]) == "both"


def test_rebuild_window_handles_two_spans_same_line():
    cfg = TrimeterNoiseConfig(recon_width=15, anchor_begin_p=0.0, anchor_end_p=0.0,
                              anchor_mid_p=0.0, min_middle_letters=3,
                              p_clean_context=1.0, p_independent=0.0, p_vertical_tear=0.0)
    stretch = _fake_stretch([40, 30, 40])
    win = build_window(stretch, 0, 3, 1)
    rng = np.random.default_rng(0)
    spans, mode = sample_damage(win, rng, cfg)
    assert len(spans) == 2
    out = rebuild_window(win, spans, rng, cfg)
    assert len(out["span_meta"]) == 2
    m1, m2 = out["span_meta"]
    # two non-overlapping M-wide mask blocks, in order
    assert m1["start"] + m1["M"] <= m2["start"]
    ids = out["input_ids"]
    assert ids[m1["start"]: m1["start"] + m1["M"]].eq(MASK_ID).all()
    assert ids[m2["start"]: m2["start"] + m2["M"]].eq(MASK_ID).all()
    # the surviving middle strip (between the two gaps) is unmasked, unchanged original text
    between = ids[m1["start"] + m1["M"]: m2["start"]]
    assert (between != MASK_ID).all()
