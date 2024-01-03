import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

disease_X_train = 'disease_X_train.txt'
disease_y_train = 'disease_y_train.txt'
disease_X_test = 'disease_X_test.txt'
disease_y_test = 'disease_y_test.txt'

with open(disease_X_train, 'r',encoding ="utf8") as f:
     disease_X_train = [x.rstrip().split(' ') for x in f.readlines()]

disease_X_train_arr = np.array(disease_X_train)

dis_X_train_arr_f = np.asarray(disease_X_train_arr, dtype=float)

with open(disease_y_train, 'r',encoding ="utf8") as f:
     disease_y_train = [x.rstrip().split(' ') for x in f.readlines()]

dis_y_train_arr = np.array(disease_y_train)

dis_y_train_arr_f = np.asarray(dis_y_train_arr, dtype=float)

with open(disease_X_test, 'r',encoding ="utf8") as f:
     disease_X_test = [x.rstrip().split(' ') for x in f.readlines()]

dis_X_test_arr = np.array(disease_X_test)

dis_X_test_arr_f = np.asarray(dis_X_test_arr, dtype=float)

with open(disease_y_test, 'r',encoding ="utf8") as f:
     disease_y_test = [x.rstrip().split(' ') for x in f.readlines()]

dis_y_test_arr = np.array(disease_y_test)

dis_y_test_arr_f = np.asarray(dis_y_test_arr, dtype=float)

mean_train = np.mean(dis_y_train_arr_f)

baseline_predictions = np.full_like(dis_y_test_arr_f, mean_train)

baseline_mse = mean_squared_error(dis_y_test_arr_f, baseline_predictions)

print()

print(f'Mean Squared Error for Baseline: {baseline_mse:.3f}')

lm = LinearRegression()

lm.fit(dis_X_train_arr_f,dis_y_train_arr_f)

lm_predict = lm.predict(dis_X_test_arr_f)

mse_linear_regression = mean_squared_error(lm_predict, dis_y_test_arr_f)

print()

print(f'Mean Squared Error for Linear Model: {mse_linear_regression:.3f}')

dt_model = DecisionTreeRegressor()

dt_model.fit(dis_X_train_arr_f,dis_y_train_arr_f)

dt_predict = dt_model.predict(dis_X_test_arr_f)

mse_dt_regressor = mean_squared_error(dt_predict, dis_y_test_arr_f)

print()

print(f'Mean Squared Error for Decission Tree Regressor: {mse_dt_regressor:.3f}')

rf_model = RandomForestRegressor(max_depth=6, random_state=0)

rf_model.fit(dis_X_train_arr_f,dis_y_train_arr_f.ravel())

rf_predict = rf_model.predict(dis_X_test_arr_f)

mse_rf_regressor = mean_squared_error(rf_predict, dis_y_test_arr_f)

print()

print(f'Mean Squared Error for Random Forest Regressor: {mse_rf_regressor:.3f}')
print()
