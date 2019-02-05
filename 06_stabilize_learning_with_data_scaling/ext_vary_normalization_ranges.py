from utils import disable_tensorflow_gpu
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import mean
from numpy import std


def get_dataset(input_scaler, output_scaler):
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    if input_scaler is not None:
        input_scaler.fit(trainX)
        trainX = input_scaler.transform(trainX)
        testX = input_scaler.transform(testX)
    if output_scaler is not None:
        trainy = trainy.reshape(len(trainy), 1)
        testy = testy.reshape(len(trainy), 1)
        output_scaler.fit(trainy)
        trainy = output_scaler.transform(trainy)
        testy = output_scaler.transform(testy)
    return trainX, trainy, testX, testy


def evaluate_model(trainX, trainy, testX, testy):
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9))
    model.fit(trainX, trainy, epochs=100, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    return test_mse


def repeated_evaluation(input_scaler, output_scaler, n_repeats=30):
    trainX, trainy, testX, testy = get_dataset(input_scaler, output_scaler)
    results = list()
    for _ in range(n_repeats):
        test_mse = evaluate_model(trainX, trainy, testX, testy)
        print('>%.3f' % test_mse)
        results.append(test_mse)
    return results


results_normalized_range1 = repeated_evaluation(MinMaxScaler(), MinMaxScaler())
results_normalized_range2 = repeated_evaluation(MinMaxScaler(feature_range=(-1, 1)), MinMaxScaler(feature_range=(-1, 1)))
results_normalized_range3 = repeated_evaluation(MinMaxScaler(feature_range=(0, .5)), MinMaxScaler(feature_range=(0, .5)))
print('normalized (0, 1): %.3f (%.3f)' % (mean(results_normalized_range1), std(results_normalized_range1)))
print('normalized (-1, 1): %.3f (%.3f)' % (mean(results_normalized_range2), std(results_normalized_range2)))
print('normalized (0, .5): %.3f (%.3f)' % (mean(results_normalized_range3), std(results_normalized_range3)))

results = [results_normalized_range1, results_normalized_range2,results_normalized_range3]
labels = ['normalized (0, 1)', 'normalized (-1, 1)', 'normalized (0, .5)']
pyplot.boxplot(results, labels=labels)
pyplot.show()
