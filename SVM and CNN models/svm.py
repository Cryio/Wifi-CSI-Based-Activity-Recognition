import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load iris dataset
iris = load_iris()

# Display feature and target names
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

# Create DataFrame and display first few rows
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("\nFirst few rows of the DataFrame:")
print(df.head())

# Add the target column and display the updated DataFrame
df['target'] = iris.target
print("\nDataFrame with target column added:")
print(df.head())

# Display first few rows where the target is 1 (Versicolor) and 2 (Virginica)
print("\nFirst few rows where target is 1 (Versicolor):")
print(df[df.target == 1].head())

print("\nFirst few rows where target is 2 (Virginica):")
print(df[df.target == 2].head())

# Add flower name column
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print("\nDataFrame with flower name column added:")
print(df.head())

print("\nRows 45 to 55 of the DataFrame:")
print(df[45:55])

# Split DataFrame into three parts
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# Plot Sepal Length vs Sepal Width (Setosa vs Versicolor) and Petal Length vs Petal Width (Setosa vs Versicolor)
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

ax[0].set_xlabel('Sepal Length')
ax[0].set_ylabel('Sepal Width')
ax[0].scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label='Setosa')
ax[0].scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label='Versicolor')
ax[0].set_title('Sepal Length vs Sepal Width (Setosa vs Versicolor)')
ax[0].legend()

ax[1].set_xlabel('Petal Length')
ax[1].set_ylabel('Petal Width')
ax[1].scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label='Setosa')
ax[1].scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label='Versicolor')
ax[1].set_title('Petal Length vs Petal Width (Setosa vs Versicolor)')
ax[1].legend()

plt.show()

# Train Using Support Vector Machine (SVM)
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"\nLength of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")

model = SVC()
model.fit(X_train, y_train)

print(f"\nModel Score: {model.score(X_test, y_test)}")
print(f"Prediction for [4.8, 3.0, 1.5, 0.3]: {model.predict([[4.8, 3.0, 1.5, 0.3]])}")

# Tune parameters
# 1. Regularization (C)
model_C = SVC(C=1)
model_C.fit(X_train, y_train)
print(f"\nModel Score with C=1: {model_C.score(X_test, y_test)}")

model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print(f"Model Score with C=10: {model_C.score(X_test, y_test)}")

# 2. Gamma
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print(f"\nModel Score with Gamma=10: {model_g.score(X_test, y_test)}")

# 3. Kernel
model_linear_kernel = SVC(kernel='linear')
model_linear_kernel.fit(X_train, y_train)
print(f"\nModel Score with Linear Kernel: {model_linear_kernel.score(X_test, y_test)}")
