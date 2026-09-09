import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wp1'))
from feedback_loop import _restore, _omitted_ranges, run_feedback_loop


def test_restore_prefix_goes_before_first_surviving_frame():
    restored, _ = _restore('frame\nbody', 'header\nframe\nbody', (1, 1))
    assert restored.index('header') < restored.index('frame')


def test_repeated_frames_are_counted_per_occurrence_and_restore_in_order():
    raw = 'test_one\nframe\nbody_one\ntest_two\nframe\nbody_two'
    compressed = 'test_one\nframe\ntest_two\nframe\nbody_two'
    assert _omitted_ranges(raw, compressed) == [(3, 3)]
    restored, _ = _restore(compressed, raw, (3, 3))
    assert restored.index('body_one') < restored.index('test_two')
    assert _omitted_ranges('frame\nframe', 'frame') == [(2, 2)]


def test_feedback_round_cap_does_not_make_a_third_call():
    calls = []
    def verify(payload, round_num):
        calls.append(round_num)
        return 'MISSING: L1-L1 reason' if round_num == 1 else 'MISSING: L3-L3 reason'
    result = run_feedback_loop('a\nb\nc', 'b', verify)
    assert calls == [1, 2]
    assert result.stop_reason == 'round_cap'


def test_overlapping_prunes_cannot_delete_unrequested_lines():
    raw = '\n'.join(str(i) for i in range(1, 31))
    result = run_feedback_loop(raw, raw, lambda _, n:
                               'USELESS: C4-C5 noise\nUSELESS: C3-C5 noise' if n == 1 else 'OK')
    assert result.final_text.splitlines() == [str(i) for i in range(1, 31) if i not in (4, 5)]
    assert any('overlaps another prune' in r for r in result.turns[0].rejected)


def test_out_of_bounds_prune_is_rejected_without_partial_deletion():
    raw = 'a\nb\nc\nd'
    result = run_feedback_loop(raw, raw, lambda *_: 'USELESS: C0-C1 noise')
    assert result.final_text == raw
    assert 'out of bounds' in result.turns[0].rejected[0]
