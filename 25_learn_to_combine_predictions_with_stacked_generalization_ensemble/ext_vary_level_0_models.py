# stacked generalization with neural net meta model on blobs dataset
from utils import disable_tensorflow_gpu
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
from matplotlib import pyplot as plt

# fit model on dataset
def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(25, input_dim=2, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# one hot encode output variable
	trainy_enc = to_categorical(trainy)
	# fit model
	model.fit(trainX, trainy_enc, epochs=500, verbose=0)
	return model

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	inputy_enc = to_categorical(inputy)
	# fit model
	model.fit(X, inputy_enc, epochs=300, verbose=0)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)

# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

acc_list = []
# fit all level 0 models
for n_members in range(2, 11):
	members = [fit_model(trainX, trainy) for _ in range(n_members)]
	# define ensemble model
	stacked_model = define_stacked_model(members)
	# fit stacked model on test dataset
	fit_stacked_model(stacked_model, testX, testy)
	# make predictions and evaluate
	yhat = predict_stacked_model(stacked_model, testX)
	yhat = argmax(yhat, axis=1)
	acc = accuracy_score(testy, yhat)
	print('Stacked test accuracy with %d 0-level models: %.3f' % (n_members, acc))
	acc_list.append(acc)

plt.plot([str(i) for i in range(2, 11)], acc_list)
plt.show()