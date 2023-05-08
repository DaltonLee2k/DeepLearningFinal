import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset into a Pandas DataFrame
data = pd.DataFrame({
    'G': [15, 17, 17, 49],
    'QBrec': ['6-9-0', '9-8-0', '10-7-0', '25-24-0'],
    'Yds': [4336, 5014, 4739, 14089],
    'TD': [31, 38, 25, 94],
    'Int': [10, 15, 10, 35],
    'Rate': [98.3, 97.7, 93.2, 96.2]
})

# Extract the input features (previous season stats)
X = data[['G', 'Yds', 'TD', 'Int', 'Rate']].values

# Extract the output feature (next season stats)
y = data['G'].shift(-1).dropna().values

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X[:-1], y)

# Predict the player's next season stats based on their previous season stats
next_season_stats = model.predict([X[-1]])
print(f"Predicted G: {next_season_stats[0]}")

