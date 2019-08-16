## Findings about combining models from multiple runs with model averaging ensemble

### Problem description

A multiclass classification problem is used to demonstrate the effect of using a model averaging ensemble to reduce the
variance in the model's predictions. Specifically, the problem consists of 3 classes, 2 input features and a dataset
size of 500, which is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">
