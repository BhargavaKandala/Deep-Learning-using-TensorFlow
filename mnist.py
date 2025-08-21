import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

print(len(x_train))