import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = r'D:\Projects\WIFI Research - Git\SVM and CNN models\tabla dataset\tabla_solo_1.0\onsMap\ajr_10_dli.csv'  # file path
data = pd.read_csv(file_path)

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Summary statistics of the dataset
print("\nSummary of the dataset:")
print(data.describe())

# Info of the dataset
print("\nInfo of the dataset:")
print(data.info())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Rename columns for easier reference
data.columns = ['feature', 'target']

# Display unique target values for understanding
print("\nUnique target values:")
print(data['target'].unique())

# Encode categorical variables if necessary
data['target'] = data['target'].astype('category').cat.codes

# Get unique target values and their corresponding categories
unique_targets = data['target'].unique()
categories = ['DHA', 'TA', 'DHI', 'GE', 'NA', 'TII', 'KI'] 

# Separate features and target
X = data[['feature']]
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nLength of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")

# Create a pipeline that scales the data then trains the SVM
pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(18, 6))

# Histogram of feature distribution by target
plt.subplot(1, 3, 1)
sns.histplot(data=data, x='feature', hue='target', multiple='stack', palette='viridis')
plt.title('Feature Distribution by Target')
plt.xlabel('Feature')
plt.ylabel('Count')
plt.legend(title='Target', labels=categories)

# Box plot of feature distribution by target
plt.subplot(1, 3, 2)
sns.boxplot(x='target', y='feature', data=data, hue='target', palette='viridis')
plt.title('Box Plot of Feature by Target')
plt.xlabel('Target')
plt.ylabel('Feature')
plt.legend(title='Target', labels=categories)

# Density plot of feature distribution by target
plt.subplot(1, 3, 3)
sns.kdeplot(data=data, x='feature', hue='target', fill=True, palette='viridis')
plt.title('Density Plot of Feature by Target')
plt.xlabel('Feature')
plt.ylabel('Density')
plt.legend(title='Target', labels=categories)

# Plot SVM decision boundary
plt.figure(figsize=(10, 6))

# Generate data for decision boundary
feature_range = np.linspace(X['feature'].min() - 1, X['feature'].max() + 1, 500).reshape(-1, 1)
decision_boundary = pipeline.decision_function(feature_range)

# Plot decision boundary
plt.scatter(X['feature'], y, c=y, cmap='viridis', s=100, alpha=0.6, edgecolors='k')
plt.plot(feature_range, decision_boundary, color='red', linestyle='--')
plt.title('SVM Decision Boundary')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend(title='Target', labels=categories)

plt.tight_layout()
plt.show()
