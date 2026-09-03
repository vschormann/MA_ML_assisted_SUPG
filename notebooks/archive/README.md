# Submitted and exploratory notebook archive

These files preserve the notebook state at thesis submission. They may contain
duplicated definitions, hidden execution state, large output cells, overwritten
variables, and paths relative to the repository root. They are evidence and
working history, not the canonical reusable implementation.

## `chapter4`

The individual MLP, GCN, GraphSAGE, GAT, and GATv2 supervised and
self-supervised runs, plus the original test-set analysis. These are reported
experiments, including negative results. They are consolidated by canonical
notebooks 03-05 and the two Chapter 4 configuration files.

## `chapter5_revised`

The combined numerical-method/data-generation notebook, separate revised MLP
and GATv2 training notebooks, and final analysis. These are reported experiments
and are consolidated by canonical notebooks 06-08. `Train_revised copy.ipynb`
is specifically the revised GATv2 experiment; its old name is retained so the
submitted artifact remains recognizable.

## `prototypes`

Early objective plots, direct optimization studies, data generation, the first
self-supervised loop, and gallery generation. Their useful functions have been
extracted into `supgml`.

## `alternative_graphs`

Dense/Performer attention, MHA, sensitivity-derived edges, globalized graph
connections, alternative feature graphs, and output-clamping experiments.
These investigate ideas mentioned in the thesis outlook but are not part of the
main reported model comparison. Their status is exploratory or inconclusive;
they should not silently feed the canonical Chapter 4 results.

No archived notebook should be edited in place. If an idea is resumed, create a
new configuration and canonical experiment with a hypothesis, dataset schema,
selection metric, and output directory.
