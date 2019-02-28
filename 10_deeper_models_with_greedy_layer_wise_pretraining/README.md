## Findings about deeper models with greedy layer-wise pretraining

### Problem description

A multiclass classification problem is used to demonstrate the effect of greedy layer-wise pretraining on the capability
to train deeper models. Specifically, the problem consists of 3 classes, 2 input features and a dataset size of 1000,
which is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">
