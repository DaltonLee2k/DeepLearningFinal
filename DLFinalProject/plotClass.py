import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
data = pd.read_csv('quarterback_stats.csv')

# Separate the data into MVP and non-MVP groups based on the binary classification column
mvp_data = data[data['MVP'] == 1]
non_mvp_data = data[data['MVP'] == 0]

# Create a scatter plot with yards on the x-axis, touchdowns on the y-axis, and different colored markers for the two groups
plt.scatter(mvp_data['Yds'], mvp_data['TD'], color='blue', label='MVP')
plt.scatter(non_mvp_data['Yds'], non_mvp_data['TD'], color='red', label='Non-MVP')

# Add axis labels and a legend
plt.xlabel('Yards')
plt.ylabel('Touchdowns')
plt.legend()

# Show the plot
plt.show()
