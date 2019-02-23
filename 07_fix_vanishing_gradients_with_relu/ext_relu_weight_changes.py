from utils import disable_tensorflow_gpu
from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy.linalg import norm
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler


class WeightsMonitor(Callback):
    def on_train_begin(self, logs={}):
        self.weight_norms = []
        for l in range(len(self.model.layers)):
            self.weight_norms.append([])
            for _ in range(len(self.model.layers[l].get_weights()[0].T)):
                self.weight_norms[l].append([])

    def on_epoch_end(self, epoch, logs={}):
        for l in range(len(self.model.layers)):
            for w in range(len(self.model.layers[l].get_weights()[0].T)):
                self.weight_norms[l][w].append(norm(self.model.layers[l].get_weights()[0].T[w], 1))


X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt)

weights_monitor = WeightsMonitor()
model.fit(trainX, trainy, epochs=500, callbacks=[weights_monitor], verbose=0)

pyplot.figure(figsize=(6.4, 6.4))
for l in range(len(weights_monitor.weight_norms)):
    pyplot.subplot(420 + (l + 1))
    for w in range(len(weights_monitor.weight_norms[l])):
        pyplot.plot(weights_monitor.weight_norms[l][w], label='node ' + str(w + 1))
    pyplot.title('layer ' + str(l + 1))
    pyplot.legend(loc='upper left', fontsize='xx-small')

pyplot.show()
