import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV into a pandas DataFrame
df = pd.read_csv('quarterback_stats.csv')

# Split the QBrec column into wins, losses, and ties
df[['Wins', 'Losses', 'Ties']] = df['QBrec'].str.split('-', expand=True)

# Convert the wins, losses, and ties columns to numeric data types
df[['Wins', 'Losses', 'Ties']] = df[['Wins', 'Losses', 'Ties']].apply(pd.to_numeric)

# Drop the original QBrec column and the MVP column
X = df.drop(['QBrec', 'MVP'], axis=1)
y = df['MVP']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a binary classifier on the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the test data
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))

# Define the new stat line with the 'Wins', 'Losses', and 'Ties' columns included
new_stat_line = [[16, '13-3-0', 4299, 48, 5, 121.5]]
new_stat_line_df = pd.DataFrame(new_stat_line, columns=['G', 'QBrec', 'Yds', 'TD', 'Int', 'Rate'])

# Split the QBrec column into wins, losses, and ties
new_stat_line_df[['Wins', 'Losses', 'Ties']] = new_stat_line_df['QBrec'].str.split('-', expand=True)

# Convert the wins, losses, and ties columns to numeric data types
new_stat_line_df[['Wins', 'Losses', 'Ties']] = new_stat_line_df[['Wins', 'Losses', 'Ties']].apply(pd.to_numeric)

# Drop the QBrec column and make the prediction
new_stat_line_df = new_stat_line_df.drop('QBrec', axis=1)
new_prediction = clf.predict(new_stat_line_df)
print('New prediction:', new_prediction)

