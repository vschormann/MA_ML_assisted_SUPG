# Continuing this work

This repository is both a research record and a post-submission refactoring.
Before extending it, distinguish the evidence used in the thesis from the
interfaces added later to make that evidence easier to read.

## What is available from a fresh clone

| Material | Location | Status |
| --- | --- | --- |
| Submitted research code and notebook outputs | `notebooks/archive/` and submission commit `aed55ec` | Historical evidence; preserved as written, including exploratory state and duplicated code |
| Maintained package | `src/supgml/` | Post-submission extraction of reusable mechanics; covered by lightweight tests |
| Guided notebook sequence | `notebooks/01_*.ipynb` through `09_*.ipynb` | Post-submission reading path; some cells run independently, while later stages are interface or analysis templates |
| Experiment descriptions | `experiments/*.json` | Machine-readable descriptions for the post-submission runners; not records of the original notebook processes |
| Published visual evidence | `gallery/`, `docs/assets/thesis-figures/`, and `docs/rendered-notebooks/` | Tracked static outputs that can be inspected without the numerical environment |
| Generated documentation site | `site/` | Local MkDocs build output; ignored and safe to regenerate |
| FEM graphs, meshes, targets, and checkpoints | `data/` | Ignored since the submitted state; not included in a fresh clone |
| New checkpoints and run summaries | `runs/` | Ignored post-submission output; not included in a fresh clone |

Consequently, a fresh clone can build the documentation, inspect the submitted
notebooks and static results, validate experiment configurations, and test the
refactored interfaces. It cannot reproduce the full thesis training runs
without separately restoring or regenerating the data and model artefacts.
There is currently no published data archive or checksum manifest.

In this repository, **canonical** means “the maintained explanatory entry
point.” It does not mean that a notebook is the exact submitted file or a
complete one-click reproduction. For the executed historical path, use the
submitted notebooks and the submission commit.

## Choose the right starting point

| Intent | Start with |
| --- | --- |
| Understand the scientific question and findings | [Results and research path](results.md), then the thesis |
| Audit exactly what was submitted | commit `aed55ec` and `notebooks/archive/README.md` |
| Understand the maintained interfaces | the [guided notebook map](notebooks.md) and tutorials |
| Reuse a numerical or ML component | [Architecture](architecture.md), then the relevant `supgml` subpackage |
| Start a new experiment | a new JSON configuration and a new notebook or analysis under a clearly named post-submission path |

Do not edit an archived notebook or historical figure in place. A new result
should be labelled as new work and should not be described as a thesis result.

## Preflight for a new run

1. Create a working DOLFINx environment as described in the
   [installation guide](installation.md), then install the ML and visualization
   extras into that same environment.
2. Restore or regenerate the required inputs under the exact paths named by the
   selected `experiments/*.json` file. Check graph feature names and order,
   schema version, mesh IDs, target and bound shapes, and the training/test
   split before training.
3. Validate the configuration without starting a long run:

   ```bash
   supgml-train experiments/ch4_supervised.json --dry-run
   ```

4. Treat the full commands as new runs of the refactored runner. Chapter 4 uses
   20,000 epochs; Chapter 5 uses 200,000 MLP epochs and 150,000 GATv2 epochs,
   and FEM-backed evaluation can be expensive.
5. Record the Git commit, environment and library versions, configuration,
   random seed, input checksums, hardware/MPI layout, checkpoint-selection
   rule, and both parameter and FEM-solution metrics.
6. Inspect solution fields and the thesis line cuts. The thesis shows that a
   small target or objective loss does not by itself rule out oscillation,
   smearing, or outflow artefacts.

## Research directions already motivated by the thesis

The following are open directions from the Chapter 4 discussion and Chapter 5
analysis, not features already implemented or validated here:

- build a larger, quality-controlled dataset using a common robust reference,
  such as an algebraically stabilized solution;
- normalize or redesign objectives so that one problem's loss scale does not
  dominate heterogeneous training;
- model interior, inflow-boundary, and outflow-boundary cells separately;
- compare cell adjacency with streamline connections, learned dense attention,
  or other long-range graph structures; and
- address non-unique optimized parameter targets by judging predictions through
  the state solution as well as by distance to one chosen target.

For any continuation, preserve negative results and qualitative failures. That
is essential here because the thesis's main conclusion is deliberately
qualified: adjacent-cell information was useful and GATv2 generalized best in
the reported comparison, but no model consistently produced physical
solutions across all problems.

## Documentation sources

The maintained MkDocs source is under `docs/`. Build it with:

```bash
python -m mkdocs build
```

The resulting `site/` directory is generated and ignored; do not edit
`site/index.html` directly. The separate root `index.html` is the tracked
project/gallery landing page.
