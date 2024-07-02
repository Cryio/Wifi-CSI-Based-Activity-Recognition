import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import seaborn as sns

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
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='flower_name', style='flower_name', ax=ax[0], palette='deep')
ax[0].set_title('Sepal Length vs Sepal Width')
ax[0].legend(loc='upper right')

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='flower_name', style='flower_name', ax=ax[1], palette='deep')
ax[1].set_title('Petal Length vs Petal Width')
ax[1].legend(loc='upper right')

plt.show()

# Train Using Support Vector Machine (SVM)
X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nLength of X_train: {len(X_train)}")
print(f"Length of X_test: {len(X_test)}")

model = SVC(kernel='linear')
model.fit(X_train, y_train)

print(f"\nModel Score: {model.score(X_test, y_test)}")
print(f"Prediction for [4.8, 3.0, 1.5, 0.3]: {model.predict([[4.8, 3.0, 1.5, 0.3]])}")

# Plot decision boundaries
def plot_decision_boundary(model, X, y, features, target_names, title):
    x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
    y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=y, palette='viridis', edgecolor='k', s=100)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(title)
    plt.legend(title='Class', labels=target_names, loc='best')

# Plot decision boundaries for sepal length vs sepal width and petal length vs petal width
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Sepal length vs Sepal width
model_sepal = SVC(kernel='linear')
model_sepal.fit(X_train[['sepal length (cm)', 'sepal width (cm)']], y_train)
plt.sca(ax[0])
plot_decision_boundary(model_sepal, X_train[['sepal length (cm)', 'sepal width (cm)']], y_train, 
                       ['sepal length (cm)', 'sepal width (cm)'], iris.target_names, 'Sepal Length vs Sepal Width Decision Boundary')

# Petal length vs Petal width
model_petal = SVC(kernel='linear')
model_petal.fit(X_train[['petal length (cm)', 'petal width (cm)']], y_train)
plt.sca(ax[1])
plot_decision_boundary(model_petal, X_train[['petal length (cm)', 'petal width (cm)']], y_train, 
                       ['petal length (cm)', 'petal width (cm)'], iris.target_names, 'Petal Length vs Petal Width Decision Boundary')

plt.show()

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
