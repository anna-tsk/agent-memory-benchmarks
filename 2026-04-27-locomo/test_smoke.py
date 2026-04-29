"""Smoke test: load the dataset, check shape, resolve evidence dia_ids.

Run: python test_smoke.py
"""

from loader import load_samples


def main() -> None:
    samples = load_samples()
    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}"

    total_qa = sum(len(s.qa) for s in samples)
    total_turns = sum(sum(len(sess.turns) for sess in s.sessions) for s in samples)
    print(f"loaded {len(samples)} conversations")
    print(f"  speakers: {[(s.speaker_a, s.speaker_b) for s in samples[:3]]} ...")
    print(f"  total sessions: {sum(len(s.sessions) for s in samples)}")
    print(f"  total turns: {total_turns}")
    print(f"  total qa: {total_qa}")

    # Spot-check: nearly every QA evidence dia_id should resolve to a turn.
    # A handful of malformed entries in the source data (e.g. "D:11:26",
    # "D30:05", out-of-range turn numbers) won't resolve — those are bugs
    # in locomo10.json itself, not in the loader. We tolerate a small number.
    unresolved: list[tuple[str, str]] = []
    for sample in samples:
        for qa in sample.qa:
            for dia_id in qa.evidence:
                if sample.turn_by_dia_id(dia_id) is None:
                    unresolved.append((sample.sample_id, dia_id))
    print(f"  unresolved evidence dia_ids: {len(unresolved)} (expect ~6)")
    if unresolved:
        for sid, dia_id in unresolved[:10]:
            print(f"    {sid}: {dia_id!r}")
    assert len(unresolved) < 20, "too many unresolved evidence dia_ids"

    # show one full QA roundtrip
    s = samples[0]
    qa = s.qa[0]
    print(f"\nexample from {s.sample_id}:")
    print(f"  Q: {qa.question}")
    print(f"  A: {qa.answer}")
    print(f"  category: {qa.category}")
    for dia_id in qa.evidence:
        turn = s.turn_by_dia_id(dia_id)
        print(f"  evidence {dia_id} ({turn.speaker}): {turn.text}")


if __name__ == "__main__":
    main()
