## Findings about configuring what to optimize with loss functions

### Regression problem description

The regression problem used to demonstrate the effect of loss functions (the ones used with regression) on the speed,
stability and performance of the learning process is contrived by means of the scikit-learn `make_regression()` function.
Specifically, the dataset consists of 1000 examples (evenly split into train and test sets), 20 input features with a
noise of 0.1.

### Alternate regression loss

The mean absolute percentage error (MAPE) loss function doesn't work well with target values close to zero. This is
because the target values appear in the denominator of the MAPE function, causing the error to take on huge values in
such cases. Therefore, the problem has been adjusted so that the target values are not standardized. Even with this
adjustment the corresponding MSE is quite bad: `train: 1466.628, test: 1763.889`. As the following learning curves also
show, the learning process converges slowly, is bumpy and has bad final performance. Hence, MAPE is not a good loss
function for this kind of data.

![](images/ext_alternate_regression_loss.png)

### Multiclass classification problem description

A multiclass classification problem is used to demonstrate the effect of loss functions on the speed, stability and
performance of the learning process. Specifically, the problem consists of 3 classes, 2 input features and a dataset size
of 1000, which is contrived using the scikit-learn `make_blobs()` function.

<img src="images/multiclass_classification_problem.png" width="420">

### Alternate classification loss

Using the categorical hinge loss results in comparable performance of `train: 0.826, test: 0.836`. The learning curves
also show that the model has converged. However, the final performances vary considerably between repeated evaluations.

![](images/ext_alternate_classification_loss.png)

### Repeated evaluation

To confirm that the categorical cross-entropy, sparse categorical cross-entropy and KL divergence loss functions are
functionally equivalent, the average of 25 evaluations of each loss function is taken. As can be seen from the following
table, the metrics for the different loss functions are very close to each other.

| Loss function                    | Train loss | Train accuracy | Test loss | Test accuracy |
|----------------------------------|-----------:|---------------:|----------:|--------------:|
| Categorical cross-entropy        | 0.403      | 0.826          | 0.433     | 0.820         |
| Sparse categorical cross-entropy | 0.399      | 0.827          | 0.428     | 0.822         |
| KL divergence                    | 0.401      | 0.826          | 0.433     | 0.820         |
