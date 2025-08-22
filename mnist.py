import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

print(len(x_train))

plt.matshow(x_train[2])

y_train[2]

print(y_train[:5])

x_train.shape

x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
print(x_train_flattened.shape)

keras.Sequential([
    keras.layers.Dense
])