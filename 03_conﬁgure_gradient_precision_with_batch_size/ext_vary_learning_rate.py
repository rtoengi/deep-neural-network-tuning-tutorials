from utils import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot


def prepare_data():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


def fit_model(trainX, trainy, testX, testy, l_rate):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation='softmax'))
    opt = SGD(lr=l_rate, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0, batch_size=1)
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.title('l_rate=' + str(l_rate), pad=-40)


trainX, trainy, testX, testy = prepare_data()
l_rates = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
for i in range(len(l_rates)):
    plot_no = 420 + (i + 1)
    pyplot.subplot(plot_no)
    fit_model(trainX, trainy, testX, testy, l_rates[i])
pyplot.show()
