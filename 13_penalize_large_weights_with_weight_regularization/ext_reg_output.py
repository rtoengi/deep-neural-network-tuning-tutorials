from utils import disable_tensorflow_gpu
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from matplotlib import pyplot

X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


def fit_model(l):
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(l)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(trainX, trainy, epochs=4000, validation_data=(testX, testy), verbose=0)
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f (lambda=%s)' % (train_acc, test_acc, str(l)))

    pyplot.subplot(211)
    pyplot.title('Cross-Entropy Loss (lambda=' + str(l) + ')', pad=-40)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.subplot(212)
    pyplot.title('Accuracy (lambda=' + str(l) + ')', pad=-40)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
    pyplot.show()


lambdas = [0, 0.001]
for l in lambdas:
    fit_model(l)
