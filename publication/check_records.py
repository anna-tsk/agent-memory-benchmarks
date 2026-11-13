#!/usr/bin/env python3
"""Offline integrity/arithmetic checks for selected publication records.

No model calls, network access, imports of experiment runners, or file writes.
This verifies preserved records, not the validity of labels or AI judgments.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'publication'


def read(name):
    return json.loads((HERE / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--locomo', type=Path, help='Optional externally obtained locomo10.json')
    args = parser.parse_args()
    manifest = read('MANIFEST.json')
    for rel, digest in manifest['sha256'].items():
        path = (ROOT / rel).resolve()
        require(ROOT in path.parents, 'Manifest path escapes release root')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Hash mismatch: ' + rel)
    rows = read('november_scores.json')['rows']
    require(len(rows) == 48 and len({r['answer_id'] for r in rows}) == 48, 'Expected 48 unique ratings')
    require(sum(r['ambiguous'] for r in rows) == 11, 'Flag total differs')
    metrics = read('november_results.json')
    for cond, saved in metrics['pooled'].items():
        rs = [r for r in rows if r['condition'] == cond]
        require(len(rs) == 24, 'Condition denominator differs')
        for cls in ['supported', 'unsupported', 'contradicted_premise', 'challenge', 'overall']:
            sub = [r for r in rs if cls == 'overall' or
                   (cls == 'challenge' and r['ground_truth_class'] != 'supported') or
                   r['ground_truth_class'] == cls]
            require(sum(r['strict_correct'] for r in sub) == saved[cls]['correct'], cond + '/' + cls)
            require(len(sub) == saved[cls]['total'], cond + '/' + cls + ' denominator')
        pairs = defaultdict(list)
        for r in rs:
            pairs[r['pair_id']].append(r['strict_correct'])
        require(len(pairs) == 12 and all(len(v) == 2 for v in pairs.values()), 'Pair structure differs')
        require(sum(all(v) for v in pairs.values()) == saved['pair_success']['correct'], 'Pair count differs')
    for cell in read('may_results.json')['cells']:
        convs = cell['conversations']
        require(sum(c['correct'] for c in convs) == cell['micro_correct'], 'May correct total')
        require(sum(c['n'] for c in convs) == cell['micro_n'], 'May denominator')
        macro = sum(c['correct'] / c['n'] * 100 for c in convs) / len(convs)
        require(abs(macro - cell['macro_pct']) < 1e-10, 'May macro average')
    diagnostic = read('retrieval_summary.json')
    require(diagnostic['unique_prespecified_required_turns'] == 16, 'Required-turn denominator')
    require(sum(diagnostic['unique_required_direct_stage_counts'].values()) == 16, 'Turn-stage accounting')
    case = read('caroline_trace.json')
    evidence = case['required_turn']
    stored = {e['name'].lower() for e in evidence['cached_entities']}
    query = {e['name'].lower() for e in case['query_extraction']['entities']}
    require(not stored.intersection(query), 'Recorded entity mismatch differs')
    require(evidence['source']['text'] == evidence['cached_raw_text'], 'Source/cache text differs')
    require(all(r['claim_id'] != evidence['cached_claim_id'] for r in case['ranked_candidates']), 'Claim found among candidates')
    require(all(r['claim_id'] != evidence['cached_claim_id'] for r in case['rendered_source_ids']), 'Claim found among rendered IDs')
    require(not evidence['links_from_selected_parents'], 'Recorded link route exists')
    if args.locomo:
        raw = args.locomo.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == read('provenance.json')['locomo_sha256'], 'LoCoMo hash differs')
        sample = next(s for s in json.loads(raw) if s['sample_id'] == case['conversation_id'])
        require(sample['qa'][87]['question'] == case['question'], 'Dataset question differs')
        turn = next(t for t in sample['conversation']['session_2'] if t['dia_id'] == 'D2:12')
        require(turn['text'] == evidence['source']['text'] and turn['speaker'] == 'Caroline', 'Dataset source differs')
    print('PASS: included hashes, saved score arithmetic, pair counts and recorded trace checks.')
    print('Not verified: human correctness, complete historical retrieval replay, or new model performance.')


if __name__ == '__main__':
    main()
