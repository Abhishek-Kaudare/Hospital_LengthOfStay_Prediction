import timeit
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
X_train=np.load('X_train.npy')
y_train=np.load('y_train.npy')
X_test=np.load('X_test.npy')
y_test=np.load('y_test.npy')
def save_pkl(model, name):
    filename = name
    pickle.dump(model, open(filename, 'wb'))
    
def test_load_pkl(name,X_test):
    loaded_model = pickle.load(open(name, 'rb'))
    result = loaded_model.predict(X_test)
    return result

start = timeit.default_timer()
from sklearn.ensemble import RandomForestRegressor
name='RandomForest_reg'
#run.tag("Description","{} Regressor".format(name))

model = RandomForestRegressor(n_estimators = 57 ,random_state = 0) # n_estimator is the # of trees built
model.fit(X_train, y_train)
filename='LOS_RF_model.pkl'.format(name)
save_pkl(model,filename)
y_pred= test_load_pkl(filename,X_test)
mse=mean_squared_error(y_test, y_pred)
print('Mean Squared Error for {} is'.format(name),mse)
