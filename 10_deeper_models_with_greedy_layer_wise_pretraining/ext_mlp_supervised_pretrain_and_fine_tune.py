from utils import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot


def prepare_data():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, testX, trainy, testy


def get_base_model(trainX, trainy):
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=100, verbose=0)
    return model


def evaluate_model(model, trainX, testX, trainy, testy):
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    return train_acc, test_acc


def add_layer(model, trainX, trainy):
    output_layer = model.layers[-1]
    model.pop()
    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(output_layer)
    model.fit(trainX, trainy, epochs=100, verbose=0)


def fine_tune(model, trainX, trainy):
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=500, verbose=0)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])


trainX, testX, trainy, testy = prepare_data()
model = get_base_model(trainX, trainy)
fine_tune(model, trainX, trainy)
scores = dict()
train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
scores[len(model.layers)] = (train_acc, test_acc)

n_layers = 10
for i in range(n_layers):
    add_layer(model, trainX, trainy)
    fine_tune(model, trainX, trainy)
    train_acc, test_acc = evaluate_model(model, trainX, testX, trainy, testy)
    print('> layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
    scores[len(model.layers)] = (train_acc, test_acc)

trainScores = [scores[k][0] for k in scores.keys()]
testScores = [scores[k][1] for k in scores.keys()]
pyplot.plot(scores.keys(), trainScores, label='train', marker='.')
pyplot.plot(scores.keys(), testScores, label='test', marker='.')
pyplot.legend()
pyplot.show()

pyplot.boxplot([trainScores, testScores], labels=['train acc', 'test acc'])
pyplot.show()
