## Findings about configuring capacity with nodes and layers

### Too many nodes using only one hidden layer

As the following learning curves of the final training error and the test accuracy show, a suitable number of nodes seem
to be somewhere between 6 and 16. With more than 16 nodes the performance of the model decreases as the model becomes
too complex for the training dataset. Interestingly, the models with 30, 34 and even 50 nodes perform well which is due
to the stochastic nature of the learning algorithm.

<img src="ext_too_many_nodes_loss.png" width="420"> <img src="ext_too_many_nodes_accuracy.png" width="420">

### Too many hidden layers

A model with up to 3 hidden layers can learn the problem perfectly. However, using more hidden layers causes the model
to decrease in performance as is illustrated by the training error and test accuracy plots below.

<img src="ext_harder_problem_loss_initial_problem.png" width="420"> <img src="ext_harder_problem_accuracy_initial_problem.png" width="420">
