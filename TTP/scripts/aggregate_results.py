"""Validate and aggregate one explicit shard group. Standard library only."""
import argparse
import csv
import json
from pathlib import Path


def aggregate(run_dir, num_shards):
    run_dir = Path(run_dir)
    if num_shards < 1:
        raise ValueError('num_shards must be positive')
    rows, reference, seen = [], None, set()
    for shard in range(num_shards):
        directory = run_dir / f'shard_{shard:02d}_of_{num_shards:02d}'
        config = json.loads((directory / 'config.json').read_text())
        result = json.loads((directory / 'results.json').read_text())
        if config['num_shards'] != num_shards or config['shard_id'] != shard:
            raise ValueError('Shard identity mismatch')
        comparable = {k: v for k, v in config.items() if k not in ('shard_id', 'gpu', 'workers', 'print_freq')}
        if reference is None:
            reference = comparable
        elif reference != comparable:
            raise ValueError('Shard configurations differ (including ensemble mode)')
        if result['ensemble_mode'] != config['ensemble_mode']:
            raise ValueError('Result/config ensemble mode mismatch')
        with (directory / 'samples.csv').open(newline='') as f:
            part = list(csv.DictReader(f))
        expected = list(range(config['dataset_size']))[shard::num_shards]
        if config['max_samples'] is not None:
            expected = expected[:config['max_samples']]
        if config.get('source_indices') is not None:
            expected = [config['source_indices'][i] for i in expected]
        indices = [int(r['index']) for r in part]
        if len(indices) != len(set(indices)) or seen.intersection(indices):
            raise ValueError('Duplicate sample indices')
        if set(indices) != set(expected) or result['evaluated'] != len(part):
            raise ValueError('Missing or unexpected sample indices/counts')
        for row in part:
            for method in ('clip', 'ttp'):
                if int(row[f'{method}_correct']) != int(row[f'{method}_pred'] == row['target']):
                    raise ValueError('Prediction/correct flag mismatch')
        seen.update(indices)
        rows.extend(part)
    if not rows:
        raise ValueError('No evaluated samples')
    rows.sort(key=lambda r: int(r['index']))
    n = len(rows)
    expected_adv = reference['eps'] > 0
    result = dict(ensemble_mode=reference['ensemble_mode'], clean_strategy=reference['clean_strategy'],
                  evaluated=n, num_shards=num_shards,
                  clip_acc=100 * sum(int(r['clip_correct']) for r in rows) / n,
                  ttp_acc=100 * sum(int(r['ttp_correct']) for r in rows) / n,
                  detection_acc=100 * sum(bool(int(r['detected_adv'])) == expected_adv for r in rows) / n,
                  config=reference)
    with (run_dir / 'samples_all_shards.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / 'aggregate.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--num-shards', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_dir, args.num_shards), indent=2))
