"""Window builder correctness: offsets tile the concatenated window, line-final boundaries are
preserved (not demoted like meter/dataset.py's concat_verses), and char-budget retries never
overshoot."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trimeter.data.corpus import Line, Stretch
from trimeter.data.window import build_window, sample_window


def _fake_line(n, length, seed):
    rng = np.random.default_rng(seed)
    chars = rng.integers(0, 24, length).astype(np.uint8)
    boundary = np.zeros(length, np.uint8)
    boundary[-1] = 2
    if length > 1:
        boundary[: length - 1][rng.random(length - 1) < 0.3] = 1
    dia = np.zeros(length, np.uint8)
    cap = np.zeros(length, np.uint8)
    punct = np.zeros(length, np.uint8)
    return Line(str(n), "x" * length, chars, boundary, dia, cap, punct)


def _fake_stretch(lengths, urn="tlg9999.tlg001"):
    lines = [_fake_line(i, L, seed=i) for i, L in enumerate(lengths)]
    return Stretch(urn, "Testus", "Πεῖρα", lines)


def test_offsets_reconstruct_original_lines():
    lengths = [20, 35, 10, 40, 15]
    stretch = _fake_stretch(lengths)
    win = build_window(stretch, 0, len(lengths), 2)
    assert win.offsets[0][0] == 0
    total = sum(lengths)
    assert win.offsets[-1][1] == total
    # tiles [0, total) with no gaps/overlap
    for (s0, e0), (s1, e1) in zip(win.offsets, win.offsets[1:]):
        assert e0 == s1
    for i, (s, e) in enumerate(win.offsets):
        assert e - s == lengths[i]
        assert np.array_equal(win.chars[s:e], stretch.lines[i].chars)
        assert np.array_equal(win.boundary[s:e], stretch.lines[i].boundary)


def test_line_boundaries_preserved_not_demoted():
    lengths = [12, 18, 22]
    stretch = _fake_stretch(lengths)
    win = build_window(stretch, 0, len(lengths), 0)
    for s, e in win.offsets:
        assert win.boundary[e - 1] == 2, "line-final boundary must stay sentence-final (2)"


def test_sample_window_respects_bounds():
    stretch = _fake_stretch([30] * 25)
    rng = np.random.default_rng(0)
    for _ in range(200):
        win = sample_window(stretch, rng, w_min=1, w_max=30, T_char=4096, safety_margin=256)
        assert win is not None
        assert 1 <= len(win.offsets) <= min(30, 25)
        assert 0 <= win.primary_idx < len(win.offsets)


def test_sample_window_respects_char_budget():
    # worst case: 30 lines of 65 letters each (corpus max observed)
    stretch = _fake_stretch([65] * 30)
    rng = np.random.default_rng(1)
    for _ in range(100):
        win = sample_window(stretch, rng, w_min=1, w_max=30, T_char=2048, safety_margin=256)
        total = win.offsets[-1][1]
        assert total <= 2048 - 256


def test_sample_window_empty_stretch_returns_none():
    stretch = Stretch("tlg0000.tlg000", "Nobody", "Κενόν", [])
    rng = np.random.default_rng(0)
    assert sample_window(stretch, rng) is None
