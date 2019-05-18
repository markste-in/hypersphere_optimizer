# Hypersphere Optimizer on ray
Optimizer that works with a hypersphere on the objective function. The evaluation of the objective function is parallelized with ray.

Simply execute hypersphere_n.py to run the optimizer on a simple problem like the "Sphere function" or the "Rastrigin function" (you may wanna adjust the dimensions).

Also works on the OpenAI gym environments like BipedalWalker-v2


Inspired by the paper [Adaptive step size random search](https://ieeexplore.ieee.org/document/1098903) by M. Schummer