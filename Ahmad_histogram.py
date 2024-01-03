import numpy as np
import matplotlib.pyplot as plt

X_train = 'X_train.txt'
y_train = 'y_train.txt'
X_test = 'X_test.txt'
y_test = 'y_test.txt'

with open(X_train, 'r',encoding ="utf8") as f:
     x_train_data = [x.rstrip().split(' ') for x in f.readlines()]

with open(y_train, 'r',encoding ="utf8") as f:
    y_train_data = [x for x in f.readlines()]


x_train_data_arr = np.array(x_train_data)

y_train_data_arr = np.array(y_train_data)

x_train_data_arr = x_train_data_arr.astype(float)
x_train_data_arr_f = np.asarray(x_train_data_arr, dtype=float)

y_train_data_arr = y_train_data_arr.astype(float)
y_train_data_arr_f = np.asarray(y_train_data_arr, dtype=float)


male_heights = x_train_data_arr_f[y_train_data_arr_f == 0][:,0]
female_heights = x_train_data_arr_f[y_train_data_arr_f == 1][:,0]

male_weights = x_train_data_arr_f[y_train_data_arr_f == 0][:,1]
female_weights = x_train_data_arr_f[y_train_data_arr_f == 1][:,1]



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(male_heights, bins=10, alpha=0.5, label='Male', color='blue')
plt.hist(female_heights, bins=10, alpha=0.5, label='Female', color='red')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()


plt.subplot(1, 2, 2)
plt.hist(male_weights, bins=10, alpha=0.5, label='Male', color='blue')
plt.hist(female_weights, bins=10, alpha=0.5, label='Female', color='red')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
