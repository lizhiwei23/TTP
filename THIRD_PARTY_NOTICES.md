# Third-party source provenance

The `clip/`, `data/`, `utils/` modules and the PGD/base-attack files in `torchattacks/` were copied from a local checkout of:

- Repository: https://github.com/TomSheng21/R-TPT
- Git HEAD: `c087c27849ba5a8e40161bac667b1fff3dd3a6ca`
- The local checkout may contain modifications beyond that commit. This identifier records the upstream base, not an assertion of a pristine checkout.

R-TPT acknowledges TPT and adversarial-attacks-pytorch. CLIP utilities also derive from OpenAI CLIP; dataset preparation follows CoOp. Existing inline copyright and attribution notices are retained.

Changes in this release include portable dataset and cache paths, a standalone TTP entry point, explicit probability/legacy-logit aggregation, full run configuration records, and checked shard aggregation. Only PGD and its attack base class are included from the vendored attack library.

No LICENSE, COPYING, or NOTICE file was present in the supplied R-TPT checkout. This package therefore does not assign a blanket MIT or other license to inherited code, and this attribution document is not a license grant. Distribution permissions and an explicit license for newly authored TTP code remain to be resolved by the repository owners before presenting the repository as fully licensed open-source software.

Upstream references:
- https://github.com/azshue/TPT
- https://github.com/openai/CLIP
- https://github.com/KaiyangZhou/CoOp
- https://github.com/Harry24k/adversarial-attacks-pytorch
