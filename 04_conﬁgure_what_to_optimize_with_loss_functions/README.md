## Findings about configuring what to optimize with loss functions

### Regression problem description

The regression problem used to demonstrate the effect of loss functions (the ones used with regression) on the speed,
stability and performance of the learning process is contrived by means of the scikit-learn `make_regression()` function.
Specifically, the dataset consists of 1000 examples (evenly split into train and test sets), 20 input features with a
noise of 0.1.
