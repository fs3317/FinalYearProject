
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
from eli5.sklearn import PermutationImportance

tf.random.set_seed(1)
np.random.seed(1)
sns.set()
plt.rcParams["font.family"] = "Times New Roman"

BOD = pd.read_excel('./outputBOD.xlsx')
NH3 = pd.read_excel('./outputNH3.xlsx', index_col = 0)
TN = pd.read_excel('./outputTN.xlsx')

# BOD MODEL SAMPLE

# SPLIT TEST AND TRAIN
X = pd.DataFrame(data=NH3, columns=["BOD", "NH3-N", "TN", "MLSS", "PH", "AT_Temp", "Date"])
y = pd.DataFrame(data=NH3, columns=["NH3_Y"])


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
        tf.keras.layers.Dense(15, activation=tf.nn.relu),
        tf.keras.layers.Dense(25, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='linear')
     ])
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model

# Wrapping the neural network model in Keras Regressor
model = KerasRegressor(build_fn=tf_model, epochs = 30)
model.fit(x_train,y_train)

perm = PermutationImportance(model, random_state=1).fit(x_train,y_train)
plt.ylim([0, 0.38])
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importance for NH3 Prediction')
M=['BOD','NH3-N','TN','MLSS','PH','AT_Temp']
importance=perm.feature_importances_
ans=sum(importance)
importance=importance/ans
print (importance)
sns.barplot(x=M,y = importance)
plt.show()

predictions = model.predict(x_test)
y_traine = model.predict(x_train)
testingerror = model.predict(x_train)

print('Mean Squared Training Error:', loss(y_true = y_train,y_pred =  y_traine))
print('Mean Squared Testing Error:',loss(y_pred = predictions, y_true = y_test))
print('R squared:',r2(y_pred = predictions, y_true = y_test))

# BOD GRAPH WITH LIMITS
figure, (ax1, ax2) = plt.subplots(1, 2)
x_label = x_test_t["Date"]
ax1.scatter(x_label, y_test)
ax2.scatter(x_label, predictions, color = 'red')
ax1.set(xlabel='Date',ylabel='BOD (mg O2/L)')
ax2.set(xlabel='Date')
ax1.set_title('Actual BOD Output')
ax2.set_title('Predicted BOD Output')
ax2.set_ylim([4, 12.3])
ax1.set_ylim([4, 12.3])
#plt.show()

fig, ax = plt.subplots()
x = np.linspace(4,12.3,100)
plt.plot(x,x,color="black")
ax.scatter(y_test, predictions,color="blue")
ax.set(xlabel='Actual BOD',ylabel='Predicted BOD')
#plt.show()

# NH3 GRAPHS WITH LIMITS
figure, (ax1, ax2) = plt.subplots(1, 2)
x_label = x_test_t["Date"]
ax1.scatter(x_label, y_test)
ax2.scatter(x_label, predictions, color = 'red')
ax1.set(xlabel='Date',ylabel='NH3-N (mg/L)')
ax2.set(xlabel='Date')
ax1.set_title('Actual NH3 Output')
ax2.set_title('Predicted NH3 Output')
ax2.set_ylim([0.45, 1.8])
ax1.set_ylim([0.45, 1.8])
plt.show()

fig, ax = plt.subplots()
x = np.linspace(0.45, 1.8,100)
plt.plot(x,x,color="black")
ax.scatter(y_test, predictions,color="blue")
ax.set(xlabel='Actual NH3',ylabel='Predicted NH3')
plt.rcParams["font.family"] = "Times New Roman"
plt.show()

figure, (ax1, ax2) = plt.subplots(1, 2)
x_label = x_test_t["Date"]
ax1.scatter(x_label, y_test)
ax2.scatter(x_label, predictions, color = 'red')
ax1.set(xlabel='Date',ylabel='TN (mg/L)')
ax2.set(xlabel='Date')
ax1.set_title('Actual TN Output')
ax2.set_title('Predicted TN Output')
ax2.set_ylim([4, 12.5])
ax1.set_ylim([4, 12.5])
#plt.show()

fig, ax = plt.subplots()
x = np.linspace(4, 12.5,100)
plt.plot(x,x,color="black")
ax.scatter(y_test, predictions,color="blue")
ax.set(xlabel='Actual TN',ylabel='Predicted TN')
#plt.show()
