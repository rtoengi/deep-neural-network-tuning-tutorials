## Findings about using models from contiguous epochs with horizontal voting ensembles

### Problem description

A multiclass classification problem is used to demonstrate the effect of using horizontal voting ensembles to improve
the predictions as well as to reduce the variance of the predictions. Specifically, the problem consists of 3 classes, 2
input features and a dataset size of 1100, 100 of which are used for training and the rest for validation and testing.
The dataset is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">

### Using smaller dataset size

When halving the dataset size the advantage of using horizontal voting ensembles becomes more apparent. The average
performance of the single models remains more or less the same, whereas the performances of the ensembles increase by
roughly 1 percent.
