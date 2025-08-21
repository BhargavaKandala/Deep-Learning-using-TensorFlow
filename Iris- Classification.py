import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

num_classes = 3
y_train_one_hot =to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

num_features = x_train.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features, ), kernel_regularizer=regularizers.l2(0.001)),
    
    BatchNormalization(),
    
    Dropout(0.3),
    
    Dense(64, activation='relu', kernel_regularizer = regularizers.l2(0.001)),
    
    BatchNormalization(),
    
    Dropout(0.3),
    
    Dense(num_classes, activation = 'softmax')
])

model.summary()

model.compile(optimizer = 'adam',loss= 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train_one_hot, epochs = 50, batch_size = 16, validation_split = 0.2, verbose = 1)

loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose = 2)
print(f"\nTest Loss: {loss: .4f}")
print(f"Test Accuracy: {accuracy: .4f}")