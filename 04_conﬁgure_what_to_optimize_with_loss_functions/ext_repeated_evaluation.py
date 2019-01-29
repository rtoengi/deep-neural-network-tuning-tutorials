from utils import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical


def prepare_data(one_hot):
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    if one_hot:
        y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def fit_model(trainX, trainy, loss):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=100, verbose=0)
    return model


losses = ['categorical_crossentropy', 'sparse_categorical_crossentropy', 'kullback_leibler_divergence']
n_eval = 25
for loss in losses:
    trainX, trainy, testX, testy = prepare_data(loss != 'sparse_categorical_crossentropy')
    train_loss_sum = train_acc_sum = test_loss_sum = test_acc_sum = 0
    for _ in range(n_eval):
        model = fit_model(trainX, trainy, loss)
        train_loss, train_acc = model.evaluate(trainX, trainy, verbose=0)
        test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
        train_loss_sum += train_loss
        train_acc_sum += train_acc
        test_loss_sum += test_loss
        test_acc_sum += test_acc

    print(loss)
    print('Train loss: %.3f, Train acc: %.3f' % (train_loss_sum / n_eval, train_acc_sum / n_eval))
    print('Test loss: %.3f, Test acc: %.3f' % (test_loss_sum / n_eval, test_acc_sum / n_eval))
