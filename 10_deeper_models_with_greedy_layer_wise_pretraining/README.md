## Findings about deeper models with greedy layer-wise pretraining

### Problem description

A multiclass classification problem is used to demonstrate the effect of greedy layer-wise pretraining on the capability
to train deeper models. Specifically, the problem consists of 3 classes, 2 input features and a dataset size of 1000,
which is contrived using the scikit-learn `make_blobs()` function.

<img src="images/problem.png" width="420">

### Fine-tuning the supervised pre-trained model

The following box and whisker plot presents statistics of the accuracy performance on the train and test sets of a model
having only been greedily layer-wise pre-trained, i.e. without having fine-tuned the model afterwards. The values in
either box are the classification accuracies as evaluated against the different numbers of layers, ranging from 2 to 12
layers. As can be seen the median classification accuracy on the train set is about 0.84, whereas on the test set this
metric is slightly below 0.825.

![](images/ext_mlp_supervised_pretrain.png)

Applying fine-tuning on the model improves the performance mainly on the train set. The pre-training phase is followed
by a supervised fine-tuning learning phase for each added layer using a smaller learning rate of 0.001 and a larger
number of epochs of 500. The box and whisker plot below shows that the median classification accuracy on the train set
has increased to almost 0.87, whereas the performance on the test set is comparable to the training process where no
fine-tuning of the model has been applied. This indicates a bit of a variance problem of the model. However, this seems
acceptable as the main first goal with a training process in general is to achieve very good results on the train set
and when having achieved that one can tackle the problem of overfitting.

![](images/ext_mlp_supervised_pretrain_and_fine_tune.png)
