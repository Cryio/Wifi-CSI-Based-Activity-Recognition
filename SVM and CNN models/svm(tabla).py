import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = r'D:\Projects\WIFI Research - Git\SVM and CNN models\tabla dataset\tabla_solo_1.0\onsMap\frk_30.csv'  # file path
data = pd.read_csv(file_path)

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Summary statistics of the dataset
print("\nSummary of the dataset:")
print(data.describe())

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

# Number of runs and list to store accuracies
num_runs = 10
accuracies_linear = []
accuracies_rbf = []

# Initialize X_train and y_train with original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

for i in range(num_runs):
    print(f"\nRun {i+1} - Length of X_train: {len(X_train)}, Length of X_test: {len(X_test)}")

    # Create a pipeline that scales the data then trains the SVM
    pipeline = make_pipeline(StandardScaler(), SVC())

    # Define parameter grid for GridSearchCV
    param_grid = {
        'svc__C': [0.1, 1.0, 10.0],  # Regularization parameter
        'svc__gamma': ['scale', 'auto'],  # Kernel coefficient
        'svc__kernel': ['linear', 'rbf']  # Kernel type
    }

    # Perform Grid Search Cross Validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the best model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")
    if grid_search.best_params_['svc__kernel'] == 'linear':
        accuracies_linear.append(accuracy)
    elif grid_search.best_params_['svc__kernel'] == 'rbf':
        accuracies_rbf.append(accuracy)

    # Update training data for next iteration
    X_train = pd.concat([X_train, pd.DataFrame(X_test)], axis=0)
    y_train = pd.concat([y_train, pd.Series(y_pred)], axis=0)

    # Get unique classes in y_test and y_pred
    unique_classes = np.unique(y_test)
    print("Unique classes in y_test:", unique_classes)

    # Specify labels parameter for classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=unique_classes, target_names=categories))

# Print average accuracies for each kernel
print("\nAverage Accuracies:")
print(f"Linear Kernel: {np.mean(accuracies_linear)}")
print(f"RBF Kernel: {np.mean(accuracies_rbf)}")

# Visualization (optional, for your original single run)
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

# Plot SVM decision boundary (optional)
plt.figure(figsize=(10, 6))
feature_range = np.linspace(X['feature'].min() - 1, X['feature'].max() + 1, 500).reshape(-1, 1)
decision_boundary = best_model.decision_function(feature_range)
plt.scatter(X['feature'], y, c=y, cmap='viridis', s=100, alpha=0.6, edgecolors='k')
plt.plot(feature_range, decision_boundary, color='red', linestyle='--')
plt.title('SVM Decision Boundary')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend(title='Target', labels=categories)

plt.tight_layout()
plt.show()