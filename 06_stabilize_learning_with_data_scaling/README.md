## Findings about stabilizing learning with data scaling

### Problem description

The regression problem used to demonstrate the effect of data scaling on the stability of the learning process is
contrived by means of the scikit-learn `make_regression()` function. Specifically, the dataset consists of 1000 examples
(evenly split into train and test sets), 20 input features with a noise of 0.1.

### Normalize target variable

Normalizing the target variable seems to perform even a little better than using standardization. Running 4 experiments
results in a mean train error of 0.002 and mean test error of 0.004 (each rounded to 3 decimals). In addition, the model
learns faster having converged already after about 5 epochs as compared to about 30 epochs when the target variable is
standardized.

![](images/ext_normalize_target_losses.png)
