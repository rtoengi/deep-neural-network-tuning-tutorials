## Findings about halting training at the right time with early stopping

### Problem description

A binary classification problem is used to demonstrate the effect of using early stopping to reduce overfitting of a
model. Specifically, the problem has 2 input features and a dataset size of 1000 with a noise of 0.2, which is contrived
using the scikit-learn `make_moons()` function.

<img src="images/problem.png" width="420">
