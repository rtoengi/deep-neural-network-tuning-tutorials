## Findings about decoupling layers with dropout

### Problem description

The binary classification problem used to demonstrate the effect of using dropout to address the problem of overfitting
is contrived by means of the scikit-learn `make_circles()` function. Specifically, the dataset consists of 100 examples
(with a 30/70 train/test split) with 2 input features and a noise of 0.1.

<img src="images/problem.png" width="420">

### Using dropout on input layer

Applying dropout only to the input layer with a retention rate of `0.9` results in accuracy performance of `train: 0.800,
test: 0.600`, which is inferior to using dropout on the hidden layer. The reason for this result might be due to the
small training dataset with only 2 input variables and 30 examples.

### Using dropout and weight constraints

When using weight constraints with a maximal value of `3` in addition to using dropout on the hidden layer accuracy
performance increases slightly from `train: 0.967, test: 0.771` to `train: 0.967, test: 0.814`.
