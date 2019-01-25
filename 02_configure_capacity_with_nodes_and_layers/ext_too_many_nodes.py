from util import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot


def create_dataset():
    X, y = make_blobs(n_samples=1000, centers=20, n_features=100, cluster_std=2)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def evaluate_model(n_nodes, trainX, trainy, testX, testy):
    n_input, n_classes = trainX.shape[1], testy.shape[1]
    model = Sequential()
    model.add(Dense(n_nodes, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_classes, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(trainX, trainy, epochs=100, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    return history, test_acc


def plot_capacity_curves(y, ylabel):
    pyplot.plot([str(x) for x in num_nodes], y, '-o')
    pyplot.xlabel('# nodes')
    pyplot.ylabel(ylabel)
    pyplot.show()


trainX, trainy, testX, testy = create_dataset()
num_nodes = list(range(2, 52, 2))
final_losses = []
accuracies = []
for n_nodes in num_nodes:
    history, result = evaluate_model(n_nodes, trainX, trainy, testX, testy)
    final_losses.append(history.history['loss'][-1])
    accuracies.append(result)
    print('nodes=%d: %.3f' % (n_nodes, result))
    pyplot.plot(history.history['loss'], label=str(n_nodes))
pyplot.legend()
pyplot.show()

plot_capacity_curves(final_losses, 'final training error')
plot_capacity_curves(accuracies, 'test accuracy')
