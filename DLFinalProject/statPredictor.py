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

# Extract the output features (next season stats)
y = data[['Yds', 'TD', 'Int', 'Rate']].shift(-1).dropna().values

# Create a linear regression model for each output feature
models = []
for i in range(y.shape[1]):
    model = LinearRegression()
    model.fit(X[:-1], y[:, i])
    models.append(model)

# Predict the player's next season stats based on their previous season stats
next_season_stats = [model.predict([X[-1]])[0] for model in models]

# Print the predicted next season stats
predicted_categories = ['Yds', 'TD', 'Int', 'Rate']
for category, prediction in zip(predicted_categories, next_season_stats):
    print(f"Predicted {category}: {prediction}")
