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
	return trainX, testX, trainy, testy


def base_autoencoder(trainX, testX):
	model = Sequential()
	model.add(Dense(10, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(2, activation='linear'))
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	model.fit(trainX, trainX, epochs=100, verbose=0)
	train_mse = model.evaluate(trainX, trainX, verbose=0)
	test_mse = model.evaluate(testX, testX, verbose=0)
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
	return model


def evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy):
	output_layer = model.layers[-1]
	model.pop()
	for layer in model.layers:
		layer.trainable = False
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['acc'])
	model.fit(trainX, trainy, epochs=100, verbose=0)
	_, train_acc = model.evaluate(trainX, trainy, verbose=0)
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	model.pop()
	model.add(output_layer)
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	return train_acc, test_acc


def add_layer_to_autoencoder(model, trainX, testX):
	output_layer = model.layers[-1]
	model.pop()
	for layer in model.layers:
		layer.trainable = False
	model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
	model.add(output_layer)
	model.fit(trainX, trainX, epochs=100, verbose=0)
	train_mse = model.evaluate(trainX, trainX, verbose=0)
	test_mse = model.evaluate(testX, testX, verbose=0)
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))


def fine_tune(model, trainX):
	for layer in model.layers:
		layer.trainable = True
	model.compile(loss='mse', optimizer=SGD(lr=0.001, momentum=0.9))
	model.fit(trainX, trainX, epochs=500, verbose=0)
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))


trainX, testX, trainy, testy = prepare_data()
model = base_autoencoder(trainX, testX)
fine_tune(model, trainX)
scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
scores[len(model.layers)] = (train_acc, test_acc)
n_layers = 5
for _ in range(n_layers):
	add_layer_to_autoencoder(model, trainX, testX)
	fine_tune(model, trainX)
	train_acc, test_acc = evaluate_autoencoder_as_classifier(model, trainX, trainy, testX, testy)
	print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
	scores[len(model.layers)] = (train_acc, test_acc)

keys = scores.keys()
pyplot.plot(keys, [scores[k][0] for k in keys], label='train', marker='.')
pyplot.plot(keys, [scores[k][1] for k in keys], label='test', marker='.')
pyplot.legend()
pyplot.show()
