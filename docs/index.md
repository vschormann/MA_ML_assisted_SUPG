# SUPG-ML

`supgml` packages the reusable numerical and machine-learning code developed
for the thesis on machine-learning-assisted optimization of SUPG parameters.

The package supports three main workflows:

1. define a singularly perturbed convection-diffusion benchmark and solve it
   with cellwise SUPG parameters;
2. convert the finite-element state into a PyTorch Geometric graph; and
3. train a model either against optimized parameters or directly through a
   differentiable FEM objective.

The research notebooks remain as experiment records. New reusable behavior
belongs in `src/supgml`, with notebooks acting as small clients of that API.
