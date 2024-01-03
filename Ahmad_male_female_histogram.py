import matplotlib.pyplot as plt
import numpy as np

male_female_X_train = 'male_female_X_train.txt'
male_female_y_train = 'male_female_y_train.txt'
male_female_X_test = 'male_female_X_test.txt'
male_female_y_test = 'male_female_y_test.txt'

with open(male_female_X_train, 'r',encoding ="utf8") as f:
     male_female_height_weight = [x.rstrip().split(' ') for x in f.readlines()]

male_female_height_weight = np.array(male_female_height_weight)

with open(male_female_y_train, 'r',encoding ="utf8") as f:
    male_female_class = [x for x in f.readlines()]

male_female_class = np.array(male_female_class)

male_female_height_weight = male_female_height_weight.astype(float)
male_female_height_weight_f = np.asarray(male_female_height_weight, dtype=float)
len(male_female_height_weight_f)


male_female_class = male_female_class.astype(float)
male_female_class_f = np.asarray(male_female_class, dtype=float)
len(male_female_class_f)

male_heights = male_female_height_weight_f[male_female_class_f == 0][:,0]
female_heights = male_female_height_weight_f[male_female_class_f == 1][:,0]

male_weights = male_female_height_weight_f[male_female_class_f == 0][:,1]
female_weights = male_female_height_weight_f[male_female_class_f == 1][:,1]

height_bins = 10
height_range = [80, 220]
weight_bins = 10
weight_range = [30, 180]

male_height_hist, male_height_bins = np.histogram(male_heights, bins=height_bins, range=height_range)
female_height_hist, female_height_bins = np.histogram(female_heights, bins=height_bins, range=height_range)

male_weight_hist, male_weight_bins = np.histogram(male_weights, bins=weight_bins, range=weight_range)
female_weight_hist, female_weight_bins = np.histogram(female_weights, bins=weight_bins, range=weight_range)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(male_heights, bins=height_bins, range=height_range, alpha=0.5, color='blue', label='Male Height')
plt.hist(female_heights, bins=height_bins, range=height_range, alpha=0.5, color='red', label='Female Height')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(male_weights, bins=weight_bins, range=weight_range, alpha=0.5, color='blue', label='Male Weight')
plt.hist(female_weights, bins=weight_bins, range=weight_range, alpha=0.5, color='red', label='Female Weight')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
