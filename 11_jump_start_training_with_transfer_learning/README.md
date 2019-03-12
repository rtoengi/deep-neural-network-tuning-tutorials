## Findings about jump-start training with transfer learning

### Problem description

A multiclass classification problem is used to demonstrate transfer learning. Specifically, the problem consists of 3
classes, 2 input features and a dataset size of 1000, which is contrived using the scikit-learn `make_blobs()` function.
Two instances of this problem are created using different seeds for the pseudorandom number generator: the one with a
seed of 1 is referred to as problem 1 and the other with a seed of 2 is referred to as problem 2.

<img src="images/problem.png" width="420">
