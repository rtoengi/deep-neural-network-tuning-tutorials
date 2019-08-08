# mlp overfit on the moons dataset
import numpy as np
from sklearn.datasets import make_moons
from keras.layers import Dense
from keras.models import Sequential
from numpy.linalg import norm
# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)
# Calculate norms of hidden layer's units
weights = model.layers[0].get_weights()[0]
norms = np.array([norm([weights[0][i], weights[1][i]]) for i in range(len(weights[0]))])
mean = np.mean(norms)
