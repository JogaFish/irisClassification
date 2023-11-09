import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split


def build_model():
    dnn_model = Sequential([
        Dense(units=100, input_shape=(4, ), activation='relu'),
        Dense(units=3, activation='softmax')
    ])
    dnn_model.compile(loss='sparse_categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
    return dnn_model


url = 'IRIS.csv'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
x = pd.read_csv(url, names=column_names)
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
categorical_cols = ['species']

x = x.drop(0)

for col in numeric_cols:
    x[col] = pd.to_numeric(x[col])

for col in categorical_cols:
    x[col].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)
    x[col] = pd.to_numeric(x[col])

y = x.pop('species')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
print(y_train.dtypes)
model = build_model()
model.fit(x_train, y_train, epochs=100)

scores_train = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy for training data " + str(scores_train[1]))

scores_test = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy for test data " + str(scores_test[1]))
