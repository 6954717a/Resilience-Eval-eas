# Runtime installation

This repository provides the Resilience EAS implementation as a PARTNR-compatible source overlay. It is not a standalone Habitat/PARTNR checkout: the upstream `third_party/` modules, datasets, simulator assets, and frozen resilience stress bank are external runtime prerequisites.

## 1. Follow the canonical installation guides

Use both upstream sources:

1. [PARTNR Planner pinned installation](https://github.com/facebookresearch/partnr-planner/blob/ddfff19f4b6c098a31edea4d19e7b75db72433c2/INSTALLATION.md)
2. [Habitat-Sim installation](https://github.com/facebookresearch/habitat-sim#installation)

The pinned PARTNR environment uses Python 3.9.2, PyTorch 2.4.1, CUDA 12.4, and Habitat-Sim 0.3.3. Match the CUDA build to the target machine and use Habitat-Sim's source-install instructions when a compatible binary is unavailable.

## 2. Create a fresh PARTNR runtime

```bash
git clone https://github.com/facebookresearch/partnr-planner.git partnr-runtime
git -C partnr-runtime checkout ddfff19f4b6c098a31edea4d19e7b75db72433c2
git -C partnr-runtime submodule sync
git -C partnr-runtime submodule update --init --recursive
```

Complete the PARTNR guide's simulator, Habitat-Lab, dataset, HSSD, OVMM, and checkpoint steps before applying this overlay.

## 3. Apply the Resilience EAS overlay

Use a fresh PARTNR checkout so the overlay operation is recoverable.

```bash
git clone https://github.com/6954717a/Resilience-Eval-eas.git resilience-eval
cp -a resilience-eval/ResilienceEvaluationLayer/. partnr-runtime/habitat_llm/
cd partnr-runtime
pip install -r ../resilience-eval/requirements.txt
pip install -e .
```

All Runbook commands are executed from `partnr-runtime/`, where the package is named `habitat_llm`. If the Code URL returns `404`, authenticate with a GitHub account that has access to this repository before cloning.

## 4. Provide external run assets

The resilience `10 × 3 × 6` case requires the project-frozen perturbation bank at the exact path below. The bank is not versioned in this repository. Until a public artifact URL and checksum are published, obtain the frozen `instruction_context_v6` bank from the project authors; the resilience run cannot be reproduced from this checkout alone.

```bash
test -f data/datasets/partnr_episodes/v0_0/val_mini.json.gz
test -f data/datasets/partnr_episodes/v0_0/val.json.gz
test -d data/hssd-hab
test -d data/objects_ovmm
test -f evaluation/perturbation_banks/instruction_context_v6/manifest.json
```

The planner model is served through an OpenAI-compatible vLLM server. vLLM is intentionally managed in a separate compatible server environment and is not pinned by this repository's `requirements.txt`.

## 5. Verify the overlay and upstream runtime

Confirm that the overlay landed and imports from the runnable package:

```bash
test -f habitat_llm/examples/planner_demo_expq1.py
test -f habitat_llm/conf/experiments/resilience_joint_qwen35.yaml
test -f habitat_llm/conf/evaluation/resilience_config.yaml
python -c "import habitat_llm.examples.planner_demo_expq1"
```

Then run the PARTNR heuristic smoke test before using an LLM:

```bash
python -m habitat_llm.examples.planner_demo \
  --config-name baselines/heuristic_full_obs.yaml \
  habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz"
```

Then continue with the [numbered Runbook](https://6954717a.github.io/resilience_evaluation_web/run.html).
