from util import disable_tensorflow_gpu
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs
from keras.callbacks import EarlyStopping
import time


def prepare_data():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def fit_model(trainX, trainy, testX, testy, n_batch, patience):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    es = EarlyStopping(patience=patience)
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=0, batch_size=n_batch, callbacks=[es])
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('batch=' + str(n_batch), pad=-40)
    return history


trainX, trainy, testX, testy = prepare_data()
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 500]
patiences = [10, 10, 10, 10, 20, 20, 20, 20]
for i in range(len(batch_sizes)):
    plot_no = 420 + (i + 1)
    pyplot.subplot(plot_no)
    tic = time.time()
    history = fit_model(trainX, trainy, testX, testy, batch_sizes[i], patiences[i])
    print('Elapsed time (s):', time.time() - tic)
    print('loss:', history.history['val_loss'][-1], ', acc:', history.history['val_acc'][-1])
pyplot.show()
