import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error as loss
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor

tf.random.set_seed(1)
np.random.seed(1)

BOD = pd.read_excel('./outputBOD.xlsx')
NH3 = pd.read_excel('./outputNH3.xlsx', index_col = 0)
TN = pd.read_excel('./outputTN.xlsx')

# BOD MODEL SAMPLE

# SPLIT TEST AND TRAIN
X = pd.DataFrame(data=TN, columns=["BOD", "NH3-N", "TN", "MLSS", "PH", "AT_Temp", "Date"])
y = pd.DataFrame(data=TN, columns=["TN_Y"])

x_data_t = X[:]
y_data_t = y[:]

del x_data_t["Date"]

# standardizing the feature variables
cols = x_data_t.columns.tolist()
scalar = StandardScaler()
x_data_t = scalar.fit_transform(x_data_t)

# splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_data_t, y_data_t, test_size=0.20, random_state=42)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(X, y, test_size=0.20, random_state=42)
y_train = np.array(y_train).reshape(-1,)
y_test = np.array(y_test).reshape(-1,)


#TENSOR FLOW MODEL- NEURAL NETWORK fitting model
def tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation=tf.nn.relu),
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='linear')
     ])
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

# Wrapping the neural network model in Keras Regressor
model = KerasRegressor(build_fn=tf_model, epochs = 135)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
y_traine = model.predict(x_train)

print('Mean Squared Training Error:', loss(y_true = y_train,y_pred =  y_traine))
print('Mean Squared Testing Error:',loss(y_pred = predictions, y_true = y_test))
print('R squared:',r2(y_pred = predictions, y_true = y_test))