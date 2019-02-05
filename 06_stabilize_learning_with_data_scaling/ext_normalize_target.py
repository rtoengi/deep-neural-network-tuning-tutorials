from utils import disable_tensorflow_gpu
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

trainy = trainy.reshape(len(trainy), 1)
testy = testy.reshape(len(trainy), 1)

scaler = MinMaxScaler()
scaler.fit(trainy)
trainy = scaler.transform(trainy)
testy = scaler.transform(testy)

model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

pyplot.title('Mean Squared Error Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
