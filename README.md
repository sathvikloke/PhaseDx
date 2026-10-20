# PhaseDx

PhaseDx investigates how imaging representations, cohort selection and acquisition geometry affect the interpretation of medical-imaging classification results.

The current October study combines retained MRI/internal-CT evidence, fixed synthetic experiments and a planned independent CQ500 examination-level comparison. **External CQ500 performance has not yet been established.** Historical README claims about what a neural network learned and prospective registration are not conclusions of the current revision; the original text is preserved in [the historical README](docs/history/README_PRE_CPU_20261020.md).

- [Current execution status and 65 task cards](docs/execution/README.md)
- [Working command interfaces](docs/execution/CLI_CONTRACT.md) and [tested environment](docs/execution/ENVIRONMENT.md)
- [Source-supported Methods and limitations](docs/study/RETAINED_METHODS.md)
- [Dataset registry](docs/execution/DATASETS.md) and [historical scope disposition](docs/study/HISTORICAL_DISPOSITION.csv)
- [Public synthetic study sources and fixed configurations](reproducibility/synthetic_b25)

The portable `phasedx_external` package separates acquisition, source inventory, clinical selection, frozen image/geometry inference, reference admission and paired scoring. The CPU smoke uses fabricated inputs and requires no clinical data or model weights:

```sh
python3.13 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements/cpu.lock.txt
.venv-cpu/bin/python -m pip install --no-deps .
.venv-cpu/bin/python -m phasedx_external smoke --out /private/new-smoke
```

Raw images, clinical labels, case-level scores and private model checkpoints are not distributed here. Task completion, scientific acceptance and author approval are separate states. A favorable external result is not required for an honest paper, and publication acceptance is not guaranteed by completing the queue.
