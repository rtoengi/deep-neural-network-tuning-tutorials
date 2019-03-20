from utils import disable_tensorflow_gpu
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from numpy.linalg import norm

X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


def fit_model(l):
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(l)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(trainX, trainy, epochs=4000, verbose=0)
    return model


lambdas = [0, 0.001]
for l in lambdas:
    model = fit_model(l)
    print('L2 norm (lambda=%s): %.3f' % (str(l), norm(model.layers[0].get_weights()[0])))
