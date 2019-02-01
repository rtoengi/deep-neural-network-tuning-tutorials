from utils import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot


def prepare_data():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def fit_model(trainX, trainy, testX, testy, lrate):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=lrate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
    print('Test acc (lr=%s): %.3f' % (lrate, history.history['acc'][-1]))
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.title('initial lrate=' + str(lrate))


trainX, trainy, testX, testy = prepare_data()
lrates = [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6]
for i in range(len(lrates)):
    plot_no = 320 + (i + 1)
    pyplot.subplot(plot_no)
    fit_model(trainX, trainy, testX, testy, lrates[i])
pyplot.show()
