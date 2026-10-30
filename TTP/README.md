# TTP: Test-Time Padding

Code for **TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models**.

Zhiwei Li, Yitian Pang, Weining Wang, Zhenan Sun, Qi Li.

TTP detects potentially adversarial inputs using the cosine similarity of CLIP image features before and after spatial padding. Detected inputs receive one instance-specific padding update followed by similarity-weighted view aggregation. Clean inputs bypass adaptation, or optionally use TPT.

## Implementation status

This release is organized in the style of [R-TPT](https://github.com/TomSheng21/R-TPT). It contains the CLIP/data utilities needed to run independently of that checkout.

- `--ensemble-mode probability` (default) implements the probability aggregation in manuscript Eq. 11.
- `--ensemble-mode legacy_logits` preserves the historical implementation's weighted-logit aggregation. Existing local experiments used the historical implementation; they do **not** validate the new default probability mode. No model experiments were rerun for this packaging release.
- Padding uses four learnable RGB borders followed by bilinear resizing to 224×224. The low-entropy fraction defaults to 0.1 (6 of 64 views); this is an implementation choice where the manuscript does not specify K. One SGD update minimizes mean per-view entropy, with learning rate 5 and initialization in [0, 10/255].
- The provided attack is PGD against the original CLIP classifier, followed by TTP evaluation. It is **not** an adaptive attack against the complete defense. No claim of adaptive robustness is implied by these commands.
- This source-only package contains no experimental results, datasets, weights, or caches. The presence of a model option does not indicate that every manuscript experiment has been reproduced.

## Prerequisites

Use Python 3.10 or later. Install a matching PyTorch/torchvision pair for your CUDA platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python main.py --help
```

Packaging checks used Python with torch 2.10.0 and torchvision 0.25.0. These checks cover imports and CPU utilities, not model execution. Evaluation requires a CUDA GPU. CLIP weights are downloaded on first model evaluation into `cache/clip`; use `--model-cache-dir` to select another location. The tokenizer vocabulary is included.

## Dataset preparation

Follow the dataset download and split instructions in [CoOp](https://github.com/KaiyangZhou/CoOp). Place datasets under a separate dataset root, not the repository's `data/` directory, which contains Python modules.

| `--test_sets` | Image directory under DATA_ROOT | Split file under DATA_ROOT |
| --- | --- | --- |
| Caltech101 | caltech-101/101_ObjectCategories | caltech-101/split_zhou_Caltech101.json |
| Pets | oxford_pets/images | oxford_pets/split_zhou_OxfordPets.json |
| Cars | stanford_cars | stanford_cars/split_zhou_StanfordCars.json |
| Flower102 | oxford_flowers/jpg | oxford_flowers/split_zhou_OxfordFlowers.json |
| Aircraft | fgvc_aircraft/images | fgvc_aircraft/images_variant_test.txt and variants.txt |
| DTD | dtd/images | dtd/split_zhou_DescribableTextures.json |
| eurosat | eurosat/2750 | eurosat/split_zhou_EuroSAT.json |
| UCF101 | ucf101/UCF-101-midframes | ucf101/split_zhou_UCF101.json |

JSON splits contain `train`, `val`, and `test` lists. Each row begins with an image path relative to the image directory above and a zero-based class label. Use the original CoOp labels/class order; do not silently substitute a new split. Aircraft uses its official variant list and test annotations. Dataset roots and split files are resolved independently of the working directory.

## Evaluation

Run from the repository root, or pass the absolute path to `main.py`. Epsilon and alpha use pixel units on the 0–255 scale. Default PGD uses epsilon 4, alpha epsilon/4, 100 steps, and random initialization. CLIP backbones such as `ViT-B/32`, `ViT-B/16`, and `ViT-L/14` use 224px views in this release.

### Clean images

```bash
python main.py /path/to/datasets --test_sets DTD -a ViT-B/32 \
  --eps 0 --steps 0 --gpu 0 --lazy-views --output_dir output_results/clean
```

### PGD images

```bash
python main.py /path/to/datasets --test_sets DTD -a ViT-B/32 -b 64 \
  --eps 4 --steps 100 --gpu 0 --ensemble-mode probability \
  --output_dir output_results/pgd
```

Append `--ensemble-mode legacy_logits` to use the historical aggregation. Detection and padding settings remain identical. Legacy mode preserves the aggregation rule; exact historical numbers additionally depend on the original splits, attack cache, random draws and runtime environment.

### Optional clean-side TPT

```bash
python main.py /path/to/datasets --test_sets DTD -a ViT-B/32 \
  --eps 0 --steps 0 --clean-strategy tpt --gpu 0 \
  --output_dir output_results/clean_tpt
```

### Sharded and multi-dataset evaluation

```bash
DATA_ROOT=/path/to/datasets GPUS=0,1 NUM_SHARDS=2 MAX_PARALLEL=2 \
  DATASET=DTD MODE=probability bash scripts/run_ttp_shards.sh

DATA_ROOT=/path/to/datasets GPUS=0 MODE=probability \
  bash scripts/run_tables.sh
```

The table runner defaults to the eight listed datasets and B/32 and B/16. Override `ARCHES`, `DATASETS`, `EPS`, `STEPS`, `MODE`, `CLEAN_STRATEGY`, `OUTPUT_DIR`, or `PYTHON_BIN` as needed. All child failures propagate. Each process sees only its assigned GPU. A nonempty run directory is rejected to avoid overwriting earlier data; use a new output root for a rerun.

## Outputs and cache

Outputs are separated by aggregation and clean strategy:

```text
OUTPUT_DIR/probability/direct/ViT-B-32/DTD/eps_4_steps_100/
  shard_00_of_01/
    config.json
    samples.csv
    results.json
```

`config.json` records all command-line settings, dataset size, sampled indices where applicable, runtime torch version, release version and a Python-source fingerprint. `results.json` contains classification and detection metrics; `elapsed_sec` is run wall time, not a controlled inference-latency benchmark. Detection accuracy on an attacked-only set is attack detection rate; on a clean-only set it is clean specificity.

Aggregate an explicit, complete shard group:

```bash
python scripts/aggregate_results.py \
  --run-dir output_results/pgd/probability/direct/ViT-B-32/DTD/eps_4_steps_100 \
  --num-shards 1
```

Aggregation rejects configuration mismatches, duplicate indices, missing samples, and incomplete shards. `--max-samples` limits samples **per shard**; random subset selection requires one shard. Use separate output roots for different seeds or settings.

Attack caches are under `cache/adversarial` by default; change them with `--adv-cache-dir`. Cache keys include dataset, architecture, split, seed and attack parameters. Use a fresh cache root when changing dataset contents, weights, text prompts, software or attack-generation settings; these are not all content-hashed. `--regen-adv` regenerates attacks and `--no-cache-adv` disables caching. Probability and legacy modes can reuse the same original-CLIP attacks when the other settings match.

## Code layout

```text
main.py             CLI entry point
 ttp/               evaluation, padding, view aggregation
 clip/              CLIP encoder, tokenizer and prompt utilities
 data/              dataset adapters, class names, AugMix
 utils/             shared utilities
 torchattacks/      vendored PGD and its base class
 scripts/           sharded runner, table runner, result aggregation
 tests/             CPU-only synthetic utility tests
```

Run packaging-level tests without data, weights or a GPU:

```bash
python -m unittest discover -s tests -v
```

## Citation and acknowledgments

Please cite the TTP manuscript by the title and authors above. Add its final bibliographic identifier when available; this release does not invent a venue, year of publication or DOI.

This implementation builds on [R-TPT](https://github.com/TomSheng21/R-TPT), [TPT](https://github.com/azshue/TPT), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), and [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source provenance and license status.
