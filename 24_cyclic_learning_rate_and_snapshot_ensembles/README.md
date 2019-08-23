## Findings about cyclic learning rate and snapshot ensembles

### Problem description

A multiclass classification problem is used to demonstrate the effect of using cyclic learning rates and snapshot
ensembles to improve the predictions as well as to reduce the variance of the predictions. Specifically, the problem
consists of 3 classes, 2 input features and a dataset size of 1100, 100 of which are used for training and the rest for
validation and testing. The dataset is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">

### Increasing maximum learning rate
Increasing the maximum learning rate to `0.1` decreased the test accuracy performances of the single models on average.
Interestingly however, the test accuracy performances of the ensembles increased slightly.
