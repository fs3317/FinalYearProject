# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.rcParams["font.family"] = "Times New Roman"
# set width of bars
barWidth = 0.25

# set heights of bars

#BOD TEST AND TRAIN
train = [2.36,0.21,0.019,0.39,0.082]
test = [1.56, 0.59, 0.064, 0.44,0.093]

#AMMONIA
#train=[0.055, 0.0031, 0.0020, 0.0020,0.093]
#test=[0.051,0.0082,0.00197,0.0023,0.097]

#TN
#train=[4.771, 0.196, 0.084, 1.066, 1.17]
#test=[4.389, 0.8515, 0.348, 1.424,1.13]

labels=['MLR','KNNR','RF','ANN', 'VQR']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train, width, label='Train MSE')
rects2 = ax.bar(x + width/2, test, width, label='Test MSE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

x = [1, 2, 3]
y1 =[0.348, 0.267, 0.129]
y2 =[0.711, 0.951, 0.843]
y3 = [0.963, 0.973,0.935]
y4 = [0.789, 0.964, 0.746]
y5=[0.818, 0.722, 0.758]

x=np.array(x).reshape(-1,)
y1=np.array(y1).reshape(-1,)
y2=np.array(y2).reshape(-1,)
y3=np.array(y3).reshape(-1,)
y4=np.array(y4).reshape(-1,)

barWidth = 0.125

# Set position of bar on X axis
r1 = np.arange(len(y1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Make the plot
plt.bar(r1, y1,  width=barWidth, label='MLR')
plt.bar(r2, y2,  width=barWidth,  label='KNNR')
plt.bar(r3, y3, width=barWidth,  label='RF')
plt.bar(r4, y4, width=barWidth, label='ANN')
plt.bar(r5, y5, width=barWidth,  label='VQR')

# Add xticks on the middle of the group bars
plt.xlabel('Effluent Predicted')
plt.ylabel('R-squared value')
plt.xticks(r3, ['BOD','NH3','TN'])

# Create legend & Show graphic
plt.legend()
plt.show()