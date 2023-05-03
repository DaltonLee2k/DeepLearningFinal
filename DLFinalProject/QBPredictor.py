import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the CSV into a pandas DataFrame
df = pd.read_csv('./quarterback_stats.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('MVP', axis=1), df['MVP'], test_size=0.2, random_state=42)

# Train a binary classifier on the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the performance of the classifier on the test data
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))

# Use the trained classifier to predict the MVP classification of new, unseen stat lines
new_stat_line = [[16,'13-3-0',4299,48,5,121.5]]
new_prediction = clf.predict(new_stat_line)
print('New prediction:', new_prediction)