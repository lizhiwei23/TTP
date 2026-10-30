"""CPU-only checks using synthetic tensors and temporary data; no model calls."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from data.datautils import build_dataset
from scripts.aggregate_results import aggregate
from ttp.ensemble import aggregate_views
from ttp.padding import FixedPadResize, TrainablePadResize


class EnsembleTests(unittest.TestCase):
    def test_paper_and_legacy_are_distinct(self):
        logits = torch.tensor([[20., 0.], [0., 2.], [0., 2.]])
        weights = torch.ones(3) / 3
        paper = aggregate_views(logits, weights)
        legacy = aggregate_views(logits, weights, 'legacy_logits')
        torch.testing.assert_close(paper, (weights[:, None] * logits.softmax(-1)).sum(0, keepdim=True))
        torch.testing.assert_close(legacy, (weights[:, None] * logits).sum(0, keepdim=True))
        self.assertEqual(paper.argmax().item(), 1)
        self.assertEqual(legacy.argmax().item(), 0)
        torch.testing.assert_close(paper.sum(), torch.tensor(1.))
        with self.assertRaises(ValueError):
            aggregate_views(logits, weights, 'unknown')

    def test_padding_shapes_and_input_gradient(self):
        image = torch.rand(2, 3, 8, 8)
        for pattern in ('zero', 'white', 'random'):
            self.assertEqual(FixedPadResize(2, pattern)(image).shape, image.shape)
        padder = TrainablePadResize(3, 8, 8, 2)
        padder(image).sum().backward()
        self.assertTrue(all(p.grad is not None for p in padder.parameters()))


class DatasetTests(unittest.TestCase):
    def test_split_resolution_without_chdir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / 'dtd/images/example'
            images.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(images / 'one.jpg')
            (root / 'dtd/split_zhou_DescribableTextures.json').write_text(
                json.dumps({'test': [['example/one.jpg', 0]]}))
            dataset = build_dataset('DTD', None, str(root), mode='test')
            image, label = dataset[0]
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label.item(), 0)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for shard in range(2):
            d = self.root / f'shard_{shard:02d}_of_02'
            d.mkdir()
            config = dict(num_shards=2, shard_id=shard, ensemble_mode='probability',
                          clean_strategy='direct', dataset_size=4, max_samples=None, eps=4,
                          source_indices=None)
            (d / 'config.json').write_text(json.dumps(config))
            (d / 'results.json').write_text(json.dumps(dict(evaluated=2, ensemble_mode='probability')))
            rows = [dict(index=i, target=0, clip_pred=1, ttp_pred=0,
                         detected_adv=1, clip_correct=0, ttp_correct=1) for i in range(shard, 4, 2)]
            with (d / 'samples.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    def test_complete(self):
        result = aggregate(self.root, 2)
        self.assertEqual(result['evaluated'], 4)
        self.assertEqual(result['ttp_acc'], 100)

    def test_missing_shard(self):
        (self.root / 'shard_01_of_02/results.json').unlink()
        with self.assertRaises(FileNotFoundError):
            aggregate(self.root, 2)

    def test_mixed_mode(self):
        p = self.root / 'shard_01_of_02/config.json'
        config = json.loads(p.read_text())
        config['ensemble_mode'] = 'legacy_logits'
        p.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'configurations differ'):
            aggregate(self.root, 2)

    def test_duplicate(self):
        p = self.root / 'shard_01_of_02/samples.csv'
        lines = p.read_text().splitlines()
        p.write_text('\n'.join(lines + [lines[1]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(self.root, 2)

    def test_missing_sample(self):
        p = self.root / 'shard_01_of_02/samples.csv'
        p.write_text('\n'.join(p.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Missing'):
            aggregate(self.root, 2)


if __name__ == '__main__':
    unittest.main()
