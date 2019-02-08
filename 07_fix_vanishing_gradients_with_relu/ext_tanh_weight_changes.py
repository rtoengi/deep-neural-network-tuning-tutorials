from utils import disable_tensorflow_gpu
from keras.callbacks import Callback
from keras.initializers import RandomUniform
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

init = RandomUniform(minval=0, maxval=1)
model = Sequential()
model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
model.add(Dense(5, activation='tanh', kernel_initializer=init))
model.add(Dense(5, activation='tanh', kernel_initializer=init))
model.add(Dense(5, activation='tanh', kernel_initializer=init))
model.add(Dense(5, activation='tanh', kernel_initializer=init))
model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

weights_monitor = WeightsMonitor()
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, callbacks=[weights_monitor], verbose=0)

_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# pyplot.subplot(211)
# pyplot.title('Cross-Entropy Loss', pad=-40)
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
#
# pyplot.subplot(212)
# pyplot.title('Accuracy', pad=-40)
# pyplot.plot(history.history['acc'], label='train')
# pyplot.plot(history.history['val_acc'], label='test')
# pyplot.legend()
# pyplot.show()

pyplot.plot(weights_monitor.weight_norms[0][0], 'b', label='layer 1')
pyplot.plot(weights_monitor.weight_norms[0][1], 'b')
pyplot.plot(weights_monitor.weight_norms[0][2], 'b')
pyplot.plot(weights_monitor.weight_norms[0][3], 'b')
pyplot.plot(weights_monitor.weight_norms[0][4], 'b')
pyplot.legend()
pyplot.show()
