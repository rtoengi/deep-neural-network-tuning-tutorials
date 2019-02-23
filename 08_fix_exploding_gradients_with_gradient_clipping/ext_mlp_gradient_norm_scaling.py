from utils import disable_tensorflow_gpu
from sklearn.datasets import make_regression
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot


def prepare_data():
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def fit_model(trainX, trainy, testX, testy, norm):
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))
    opt = SGD(lr=0.01, momentum=0.9, clipnorm=norm)
    model.compile(loss='mean_squared_error', optimizer=opt)
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('norm='+str(norm), pad=-20)


trainX, trainy, testX, testy = prepare_data()
norms = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
for i in range(len(norms)):
    pyplot.subplot(3, 2, i + 1)
    fit_model(trainX, trainy, testX, testy, norms[i])

pyplot.show()
