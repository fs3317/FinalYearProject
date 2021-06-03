
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as loss
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score as r2

BOD = pd.read_excel('./outputBOD.xlsx')
NH3 = pd.read_excel('./outputNH3.xlsx')
TN = pd.read_excel('./outputTN.xlsx')

sns.set()
plt.rcParams["font.family"] = "Times New Roman"

# BOD
bod_x=pd.DataFrame(data=BOD,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp", "Date"])
bod_y=pd.DataFrame(data=BOD,columns=["BOD_Y"])

x_data_t = bod_x[:]
y_data_t = bod_y[:]

del x_data_t["Date"]

x_train, x_test, y_train, y_test = train_test_split(x_data_t, y_data_t, test_size = 0.20, random_state = 42)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(bod_x, bod_y, test_size = 0.20, random_state = 42)

x_test=sm.add_constant(x_test)
x_train= sm.add_constant(x_train)
model=sm.OLS(y_train, x_train).fit()
predictions=model.predict(x_test)
trainpred=model.predict(x_train)
print(loss(predictions, y_test), 'test')
print(loss(trainpred, y_train), 'train')
print(r2(y_test, predictions))
print(model.summary())

figure, (ax1, ax2) = plt.subplots(1, 2)
x_label = x_test_t["Date"]
ax1.scatter(x_label, y_test)
ax2.scatter(x_label, predictions, color = 'red')
ax1.set(xlabel='Date',ylabel='BOD (mgO2/L)')
ax2.set(xlabel='Date')
ax1.set_title('Actual BOD Output')
ax2.set_title('Predicted BOD Output')
ax2.set_ylim([3.5, 12.3])
ax1.set_ylim([3.5, 12.3])
plt.show()

fig, ax = plt.subplots()
x = np.linspace(4,12.3,100)
plt.plot(x,x,color="black")
ax.scatter(y_test, predictions,color="blue")
ax.set(xlabel='Actual BOD',ylabel='Predicted BOD')
plt.show()

#NH3
nh3_x=pd.DataFrame(data=NH3,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp", "Date"])
nh3_y=pd.DataFrame(data=NH3,columns=["NH3_Y"])

x_data_t = nh3_x[:]
y_data_t = nh3_y[:]

del x_data_t["Date"]

x_train, x_test, y_train, y_test = train_test_split(x_data_t, y_data_t, test_size = 0.20, random_state = 42)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(nh3_x, nh3_y, test_size = 0.20, random_state = 42)

x_test=sm.add_constant(x_test)
x_train= sm.add_constant(x_train)
model=sm.OLS(y_train, x_train).fit()
predictions=model.predict(x_test)
print(model.summary())
print(r2(y_test, predictions))
print(loss(predictions, y_test), 'test')

figure, (ax1, ax2) = plt.subplots(1, 2)
x_label = x_test_t["Date"]
ax1.scatter(x_label, y_test)
ax2.scatter(x_label, predictions, color = 'red')
ax1.set(xlabel='Date',ylabel='NH3 (mg/L)')
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

#TN
tn_x=pd.DataFrame(data=TN,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp", "Date"])
tn_y=pd.DataFrame(data=TN,columns=["TN_Y"])

x_data_t = tn_x[:]
y_data_t = tn_y[:]

del x_data_t["Date"]

x_train, x_test, y_train, y_test = train_test_split(x_data_t, y_data_t, test_size = 0.20, random_state = 42)
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(tn_x, tn_y, test_size = 0.20, random_state = 42)

x_test=sm.add_constant(x_test)
x_train= sm.add_constant(x_train)
model=sm.OLS(y_train, x_train).fit()
predictions=model.predict(x_test)
print(model.summary())
print(r2(y_test, predictions))
print(loss(predictions, y_test), 'test')

plt.rcParams["font.family"] = "Times New Roman"

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
plt.show()

fig, ax = plt.subplots()
x = np.linspace(4, 12.5,100)
plt.plot(x,x,color="black")
ax.scatter(y_test, predictions,color="blue")
ax.set(xlabel='Actual TN',ylabel='Predicted TN')
plt.show()
