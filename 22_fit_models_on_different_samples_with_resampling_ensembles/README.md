## Findings about fitting models on different samples with resampling ensembles

### Problem description

A multiclass classification problem is used to demonstrate the effect of using resampling ensembles to improve the
predictions as well as to reduce the variance of the predictions. Specifically, the problem consists of 3 classes, 2
input features and a dataset size of 5000, which is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">
