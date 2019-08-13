## Findings about decoupling layers with dropout

### Problem description

The binary classification problem used to demonstrate the effect of using dropout to address the problem of overfitting
is contrived by means of the scikit-learn `make_circles()` function. Specifically, the dataset consists of 100 examples
(with a 30/70 train/test split) with 2 input features and a noise of 0.1.

<img src="images/problem.png" width="420">
