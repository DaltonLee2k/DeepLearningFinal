import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Read in the CSV file
data = pd.read_csv('quarterback_stats.csv')

# Separate the data into MVP and non-MVP groups based on the binary classification column
mvp_data = data[data['MVP'] == 1]
non_mvp_data = data[data['MVP'] == 0]

# Create a scatter plot with yards on the x-axis, touchdowns on the y-axis, and different colored markers for the two groups
plt.scatter(mvp_data['Yds'], mvp_data['TD'], color='blue', label='MVP')
plt.scatter(non_mvp_data['Yds'], non_mvp_data['TD'], color='red', label='Non-MVP')

# Fit logistic regression model to the data
X = data[['Yds', 'TD']].values
y = data['MVP'].values
clf = LogisticRegression(random_state=0).fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 100, X[:, 0].max() + 100
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10),
                     np.arange(y_min, y_max, 1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5)

# Add axis labels and a legend
plt.xlabel('Yards')
plt.ylabel('Touchdowns')
plt.legend()

# Show the plot
plt.show()
