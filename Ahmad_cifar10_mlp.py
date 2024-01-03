import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def unpickle(Exercise_5_data):
    with open(Exercise_5_data, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

data_dict_bt_1 = unpickle('data_batch_1')

print(data_dict_bt_1.keys())

batch_1_dt = list(data_dict_bt_1[b'data'])
batch_1_lvl = list(data_dict_bt_1[b'labels'])

data_dict_bt_2 = unpickle('data_batch_2')

batch_2_dt = list(data_dict_bt_2[b'data'])
batch_2_lvl = list(data_dict_bt_2[b'labels'])

data_dict_bt_3 = unpickle('data_batch_3')
data_dict_bt_4 = unpickle('data_batch_4')
data_dict_bt_5 = unpickle('data_batch_5')

batch_3_dt = list(data_dict_bt_3[b'data'])
batch_3_lvl = list(data_dict_bt_3[b'labels'])
batch_4_dt = list(data_dict_bt_4[b'data'])
batch_4_lvl = list(data_dict_bt_4[b'labels'])
batch_5_dt = list(data_dict_bt_5[b'data'])
batch_5_lvl = list(data_dict_bt_5[b'labels'])

all_batch_dt = batch_1_dt + batch_2_dt + batch_3_dt + batch_4_dt + batch_5_dt
all_batch_lvl = batch_1_lvl + batch_2_lvl + batch_3_lvl + batch_4_lvl + batch_5_lvl

all_batch_data_arr = np.array(all_batch_dt)
all_batch_level_arr = np.array(all_batch_lvl)

data_dict_bt_test = unpickle('test_batch')
batch_test_dt = list(data_dict_bt_test[b'data'])
batch_test_lvl = list(data_dict_bt_test[b'labels'])

test_batch_dt_arr = np.array(batch_test_dt)
test_batch_lvl_arr = np.array(batch_test_lvl)

x_train_batch = all_batch_data_arr
y_train_batch = all_batch_level_arr

x_train_norm = x_train_batch/255.0

x_test = test_batch_dt_arr
y_test = test_batch_lvl_arr

x_test_norm = x_test/255.0
"""
I have tried to use different hidden layer with sigmoid function and different number of neurons from neuron number 10-256 but the following hidden layer give me the best accuracy.
"""
model = keras.Sequential([
    layers.Dense(128, input_dim = 3072, activation='relu'),    # First hidden layer with 128 neurons and ReLU activation
    layers.Dense(64, input_dim = 3072, activation='relu'),     # Second hidden layer with 64 neurons and ReLU activation
    layers.Dense(32, input_dim = 3072, activation='relu'),     # Third hidden layer with 32 neurons and ReLU activation
    layers.Dense(10, input_dim = 3072, activation='softmax'),  # Output layer with 10 neurons (CIFAR-10 classes) and softmax activation
    #layers.Dense(10, input_dim=3072, activation='sigmoid')
])
# used different learning rate from 0.5 to 0.001 but 0.001 gives the best accuracy
keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

#m_hist = model.fit(x_train_norm, y_train_batch, epochs=10, batch_size=128, validation_split = 0.2)
#m_hist = model.fit(x_train_norm, y_train_batch, epochs=20, batch_size=128, validation_split = 0.3)

#the best fitted model is used
m_hist = model.fit(x_train_norm, y_train_batch, epochs=30, batch_size=128, validation_split = 0.3)

test_loss, test_acc = model.evaluate(x_train_norm, y_train_batch)



print(f'Training data acuraccy: {test_acc*100:.2f}%')

# Evaluating test data acuracy

test_loss, test_acc_1 = model.evaluate(x_test_norm, y_test)
print(f'Test data acuraccy: {test_acc_1*100:.2f}%')

plt.plot(m_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training'],loc='upper right')
plt.show()