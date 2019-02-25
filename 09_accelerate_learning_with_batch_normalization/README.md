## Findings about accelerating learning with batch normalization

### Problem description

The binary classification problem used to demonstrate the effect of batch normalization on the stability and speed of
learning is contrived by means of the scikit-learn `make_circles()` function. Specifically, the dataset consists of 1000
examples (evenly split into train and test sets), 2 input features with a noise of 0.1.

<img src="images/problem.png" width="420">
