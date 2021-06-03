
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path = "Dataset.xlsx"
sns.set()
plt.rcParams["axes.labelsize"] = 15
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
        self.tn = self.bod.rename('TN_Y')
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

# DATA CLEANER TO MAKE THE DATA SUITABLE FOR OVERSAMPLING

class DataCleaner():

    def __init__(self, df, c):
        self.df = df
        global col
        col = c
        self.rem_dup()
        self.rem_nan()
        self.rem_low()

    def rem_dup(self):
        x = self.df.drop(col, axis=1, inplace=False)
        y = pd.DataFrame(self.df[col])
        self.x = x
        self.y = y

    def rem_nan(self):
        ind = self.y[col].notna()
        self.x = self.x[ind] * 10
        self.y = self.y[ind] * 10

    def rem_low(self):
        s = self.y[col].value_counts().gt(4)
        ind = self.y[col].isin(s[s].index)

        self.x = self.x[ind]
        self.y = self.y[ind]
        self.x = self.x.fillna(0)

dc = DataCleaner(df1, 'BOD_Y')
dc.y['BOD_Y'].value_counts()
x = dc.x
y = dc.y

xr, yr = SMOTE(k_neighbors=3).fit_resample(x, y)
xr = xr/10
yr = yr/10

yr['BOD_Y'].value_counts()

N=100

xy_data = pd.DataFrame(data=df1,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp","BOD_Y", "Date"])
xy_data = xy_data.dropna()

counts=xy_data['BOD_Y'].value_counts()

new_xy_data_1 = xy_data.loc[xy_data['BOD_Y'] != counts.index[0]]
new_xy_data_2 = xy_data.loc[xy_data['BOD_Y'] == counts.index[0]]
new_xy_data_2 = new_xy_data_2.head(N)
new_xy_data=pd.concat([new_xy_data_1, new_xy_data_2], ignore_index=True)

counts=new_xy_data['BOD_Y'].value_counts()
print(counts)

for i in range(1,len(counts.index)):
  df_to_add=new_xy_data.loc[new_xy_data['BOD_Y'] == counts.index[i]]
  n=round(10/counts.values[i])
  for j in range(n):
      new_xy_data=pd.concat([new_xy_data, df_to_add], ignore_index=True)

print(new_xy_data)
new_xy_data.to_excel("outputBOD.xlsx")

x_data=pd.DataFrame(data=new_xy_data,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp", "Date"])
y_data=pd.DataFrame(data=new_xy_data,columns=["BOD_Y"])

counts=new_xy_data['BOD_Y'].value_counts()
print(counts)

N=100

xy_data = pd.DataFrame(data=df2,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp","NH3_Y", "Date"])
xy_data = xy_data.dropna()

counts=xy_data['NH3_Y'].value_counts()

new_xy_data_1 = xy_data.loc[xy_data['NH3_Y'] != counts.index[0]]
new_xy_data_2 = xy_data.loc[xy_data['NH3_Y'] == counts.index[0]]
new_xy_data_2 = new_xy_data_2.head(N)
new_xy_data=pd.concat([new_xy_data_1, new_xy_data_2], ignore_index=True)

counts=new_xy_data['NH3_Y'].value_counts()
print(counts)

for i in range(1,len(counts.index)):
  df_to_add=new_xy_data.loc[new_xy_data['NH3_Y'] == counts.index[i]]
  n=round(10/counts.values[i])
  for j in range(n):
      new_xy_data=pd.concat([new_xy_data, df_to_add], ignore_index=True)

x_data=pd.DataFrame(data=new_xy_data,columns=["BOD", "NH3-N","TN","MLSS","PH","AT_Temp", "Date"])
y_data=pd.DataFrame(data=new_xy_data,columns=["NH3_Y"])

counts=new_xy_data['NH3_Y'].value_counts()

print(new_xy_data)
new_xy_data.to_excel("outputNH3final.xlsx")

