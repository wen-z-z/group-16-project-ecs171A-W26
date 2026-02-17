import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import time as t
from sklearn.decomposition import PCA as SklearnPCA



EXAMPLES_DISPLAYED = 10

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

###PRE_PROCESS###
# Normalize the data between 1 and 0
X = X / 255.0

#convert to numpy arrays for better processing
X = X.to_numpy().astype('float32')
y = y.to_numpy().astype(int)


def train(data, labels, layer_size, model_name):
  curr_time = t.time()
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

  # Train a Multi-Layer Perceptron classifier model.
  clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=layer_size, random_state=1, max_iter = 3000)
  clf.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = clf.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

  # Generate a classification report
  print(f"~~~~~~~~~~~~{model_name}~~~~~~~~~~~~\nModel Accuracy: {accuracy:.2f}\nMSE:{mse:.2f}\nR2:{r2:.2f}")
  print(classification_report(y_test, y_pred))
  print(f"Runtime: {t.time()-curr_time:.2f} seconds\n\n")


def PCA(variance_threshold):
  #calculate eigenvectors and eigenvalues
  X_centered = X - X.mean(axis=0, keepdims=True)
  U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
  eigenvalues = (S**2) / (X.shape[0] - 1)

  #find the eigenvalues with the highest variance
  k = np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), variance_threshold) + 1

  #Vt is principal components
  #save the first k columns
  Vk = Vt[:k]
  #Z is compressed representation
  #Each sample is a k-dimensional vector
  return X_centered @ Vk.T

print("Computing explained variance...")#plot start

pca_full = SklearnPCA()

pca_full.fit(X)

explained_var = np.cumsum(pca_full.explained_variance_ratio_) 
plt.figure(figsize=(8, 5))
plt.plot(explained_var, linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axhline(y=0.80, color='g', linestyle='--', label='80% variance')

k_95 = np.argmax(explained_var >= 0.95) + 1
k_80 = np.argmax(explained_var >= 0.80) + 1

plt.axvline(x=k_95, color='r', linestyle=':')
plt.axvline(x=k_80, color='g', linestyle=':')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance on MNIST")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Components for 95% variance: {k_95}")
print(f"Components for 80% variance: {k_80}")
#p end

Z = PCA(0.95)
print(f"{X.shape}{Z.shape}")
Z2 = PCA(0.80)

train(X ,y,(16, 10), "2-layer Perceptron")
train(Z ,y,(16, 10), "2-layer Perceptron with PCA; top 95% variance")
train(Z2,y,(16, 10), "2-layer Perceptron with PCA; top 80% variance")
