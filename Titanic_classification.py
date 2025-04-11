import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np
import itertools

# Loading the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic = pd.read_csv(url) 

# EXPLORATORY DATA ANALYSIS (EDA)  

# Display the first few rows of the dataset
print(titanic.head())

# Check for missing values          
print(titanic.isnull().sum())

# Visualize survival by sex
sns.countplot(x='Sex', hue='Survived', data=titanic)
plt.title('Survival by Sex')
plt.show()

# Visualize survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic)
plt.title('Survival by Pclass')
plt.show()

# Visualize survival by AgeGroup
titanic['AgeGroup'] = pd.cut(titanic['Age'], bins=[0, 18, 35, 50, 80], labels=['0-18', '19-35', '36-50', '50+'])
sns.countplot(x='AgeGroup', hue='Survived', data=titanic)
plt.title('Survival by Age Group')
plt.show()

# FEATURE ENGINEERING

# Fill missing values using the recommended method
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# Extract titles from names with additional checks
def extract_title(name):
    if ',' in name and '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

titanic['Title'] = titanic['Name'].apply(extract_title)

# Drop the 'Name' column
titanic.drop(['Name'], axis=1, inplace=True)

# BUILDING AND TRAINING MODEL

# Encode categorical features
titanic_encoded = pd.get_dummies(titanic, columns=['Sex', 'Title', 'AgeGroup'], drop_first=True)

# Separate features and target variable
X = titanic_encoded.drop('Survived', axis=1)
y = titanic_encoded['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Making predictions
y_pred = grid_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
best_model = grid_search.best_estimator_.named_steps['classifier']
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot the feature importances
plt.figure(figsize=(10, 7))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
