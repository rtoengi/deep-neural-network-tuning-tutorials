# mlp overfit on the two circles dataset with hidden layer noise (alternate)
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GaussianNoise
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


def eval_model(stddev):
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='relu'))
    model.add(GaussianNoise(stddev))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=4000, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Stddev: %s -> Train: %.3f, Test: %.3f' % (stddev, train_acc, test_acc))


stddevs = [0.01, 0.03, 0.1, 0.3, 1.0]
for stddev in stddevs:
    eval_model(stddev)
