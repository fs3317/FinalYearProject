from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

path = "Dataset.xlsx"
sns.set()
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["font.family"] = "Times New Roman"
file = pd.ExcelFile(path)
dfs = {sheet_name: file.parse(sheet_name)
          for sheet_name in file.sheet_names}
dfs.keys()


class DataCreator:

    def __init__(self, path):
        file = pd.ExcelFile(path)
        dfs = {sheet_name: file.parse(sheet_name)
               for sheet_name in file.sheet_names}
        self.dfs = dfs
        colsa = ['BOD', 'NH3-N', 'PH', 'TN', 'Date']  # ss avg columns
        colsb = ['MLSS', 'AT_Temp']  # at avg columns
        self.df = pd.concat([dfs['SS(Ave)'][colsa], dfs['AT(Ave)'][colsb]], axis=1)
        self.odf = dfs['FE'][['BOD', "NH3-N", 'TN']]

    def get_bod(self):
        dfs = self.dfs
        self.bod = dfs['FE']['BOD']
        self.bod = self.bod.rename('BOD_Y')
        return self.bod
        # display(self.bod)

    def get_nh3(self):
        dfs = self.dfs
        self.nh3 = dfs['FE']['NH3-N']
        self.nh3 = self.nh3.rename('NH3_Y')
        return self.nh3
        # display(self.nh3)

    def get_tn(self):
        dfs = self.dfs
        self.tn = dfs['FE']['TN']
        self.tn = self.tn.rename('TN_Y')
        return self.tn
        # display(self.tn)

    def first_df(self):
        return pd.concat([self.df, self.get_bod()], axis=1)

    def second_df(self):
        return pd.concat([self.df, self.get_nh3()], axis=1)

    def third_df(self):
        return pd.concat([self.df, self.get_tn()], axis=1)


dc = DataCreator(path)
dc.df
ind = dc.get_bod().notna()
ind
dc.odf[ind]

df1 = dc.first_df()
df2 = dc.second_df()
df3 = dc.third_df()

df1=df1.dropna()
df2=df2.dropna()
df3=df3.dropna()

x_data=pd.DataFrame(data=df1,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp"])
y_data=pd.DataFrame(data=df1,columns=["BOD_Y"])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20, random_state = 42)
y_train = np.array(y_train).reshape(-1,)
y_test = np.array(y_test).reshape(-1,)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Import tools needed for visualization
feature_list = list(x_data.columns)
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='BODtree.dot', feature_names=feature_list, rounded = True, precision = 1)
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(x_data, y_data)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'BODsmall_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

plt.ylabel('Importance'); plt.xlabel('Variable');
M=['BOD','NH3-N', 'TN','MLSS','PH','AT_Temp']
sns.barplot(x=M, y=importances)
plt.ylim([0, 0.4])
plt.show()

x_data=pd.DataFrame(data=df2,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp"])
y_data=pd.DataFrame(data=df2,columns=["NH3_Y"])
y_train = np.array(y_train).reshape(-1,)
y_test = np.array(y_test).reshape(-1,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20, random_state = 42)

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Import tools needed for visualization
feature_list = list(x_data.columns)
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='NH3tree.dot', feature_names=feature_list, rounded = True, precision = 1)
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(x_data, y_data)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'NH3small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

plt.ylabel('Importance'); plt.xlabel('Variable')
M=['BOD','NH3-N', 'TN','MLSS','PH','AT_Temp']
plt.ylim([0, 0.4])
sns.barplot(x=M, y=importances)
plt.show()

x_data=pd.DataFrame(data=df3,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp"])
y_data=pd.DataFrame(data=df3,columns=["TN_Y"])
y_train = np.array(y_train).reshape(-1,)
y_test = np.array(y_test).reshape(-1,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20, random_state = 42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Import tools needed for visualization
feature_list = list(x_data.columns)
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='TNtree.dot', feature_names=feature_list, rounded = True, precision = 1)
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(x_data, y_data)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'TNsmall_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

plt.ylabel('Importance'); plt.xlabel('Variable')
M=['BOD','NH3-N', 'TN','MLSS','PH','AT_Temp']
sns.barplot(x=M, y=importances)
plt.ylim([0, 0.4])
plt.show()