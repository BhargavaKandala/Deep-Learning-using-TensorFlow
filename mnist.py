import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

print(len(X_train))

plt.matshow(X_train[2])

y_train[2]

print(y_train[:5])

print(X_train.shape)

X_train = X_train / 255
X_test = X_test / 255

X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)
print(X_train_flattened.shape)

model = keras.Sequential([
    keras.layers.Dense(10, activation="sigmoid")
])
model.compile(
    optimizer = "adam",
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_flattened,y_train, epochs = 5)

model.evaluate(X_test_flattened, y_test)

plt.matshow(X_test[1])
plt.show()

y_predicted = model.predict(X_test_flattened)
print(y_predicted[1])
print(np.argmax(y_predicted[1]))