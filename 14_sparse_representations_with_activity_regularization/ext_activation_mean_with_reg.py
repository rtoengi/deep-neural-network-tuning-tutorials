# mlp overfit on the two circles dataset
import numpy as np
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential, Model
from keras.regularizers import l1
from keras.layers import Activation
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.1, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='linear', activity_regularizer=l1(0.0001)))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)

activation_model = Model(inputs=model.input, outputs=model.layers[1].output)
activations = activation_model.predict(testX)
print('Mean activation: %f, Proportion of zero activations: %f' % (np.mean(activations), (activations == 0).sum() / activations.size))
