import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, f1_score
import time as t
from sklearn.decomposition import PCA as SklearnPCA

EXAMPLES_DISPLAYED = 10

print("[DEBUG] Loading MNIST from OpenML...")
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
X, y = mnist['data'], mnist['target']
print("[DEBUG] Loaded. Converting/normalizing...")

###PRE_PROCESS###
# Normalize the data between 1 and 0
X = X / 255.0

#convert to numpy arrays for better processing
X = X.to_numpy().astype('float32')
y = y.to_numpy().astype(int)
print(f"[DEBUG] X dtype={X.dtype}, shape={X.shape} | y dtype={y.dtype}, shape={y.shape}")

# Split once so PCA can be fit on train only and so each model uses the same split
print("[DEBUG] Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[DEBUG] Split done. X_train={X_train.shape}, X_test={X_test.shape}")


def train(X_train, X_test, y_train, y_test, layer_size, model_name, max_iter=200):
  # use perf_counter for more accurate timing
  start_fit = t.perf_counter()
  print(f"[DEBUG] Starting fit: {model_name} ...")

  # Train a Multi-Layer Perceptron classifier model.
  # early_stopping makes it stop when validation score stops improving
  clf = MLPClassifier(
      solver='adam',
      alpha=1e-5,
      hidden_layer_sizes=layer_size,
      random_state=1,
      max_iter=max_iter,
      early_stopping=True,
      n_iter_no_change=10,
      validation_fraction=0.1,
      tol=1e-4
  )
  clf.fit(X_train, y_train)
  fit_time = t.perf_counter() - start_fit

  # Make predictions on the test set
  start_pred = t.perf_counter()
  y_pred = clf.predict(X_test)
  pred_time = t.perf_counter() - start_pred

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  r2 = 1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
  # Macro F1 treats each digit class equally
  macro_f1 = f1_score(y_test, y_pred, average='macro')

  # Generate a classification report
  print(f"~~~~~~~~~~~~{model_name}~~~~~~~~~~~~")
  print(f"Input dims: {X_train.shape[1]}")
  print(f"Model Accuracy: {accuracy:.4f}")
  print(f"Macro F1:{macro_f1:.4f}")
  print(f"MSE:{mse:.4f}")
  print(f"R2:{r2:.4f}")
  print(f"Fit time: {fit_time:.2f}s | Predict time: {pred_time:.2f}s | Total: {fit_time + pred_time:.2f}s")
  print(classification_report(y_test, y_pred))
  print("\n")

  # return clf + metrics so we can later do pruning
  return clf, {
      "model": model_name,
      "input_dim": X_train.shape[1],
      "accuracy": accuracy,
      "macro_f1": macro_f1,
      "mse": mse,
      "fit_s": fit_time,
      "pred_s": pred_time,
      "total_s": fit_time + pred_time
  }


print("Computing explained variance...")  # plot start
plot_fit_start = t.perf_counter()

# Fit explained variance on full train set
pca_full = SklearnPCA(svd_solver="randomized", random_state=0)

print(f"[DEBUG] PCA plot fit: fitting on FULL X_train {X_train.shape} ...")
pca_full.fit(X_train)
print(f"[DEBUG] PCA plot fit done in {t.perf_counter() - plot_fit_start:.2f}s")

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
plt.title("PCA Explained Variance on MNIST (fit on TRAIN)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Components for 95% variance: {k_95}")
print(f"Components for 80% variance: {k_80}")
# p end


# Fit PCA once for model inputs (at k_95), then slice to k_80
print(f"[DEBUG] PCA model fit: fitting PCA once with n_components=k_95={k_95} on FULL X_train {X_train.shape} ...")
pca_model_start = t.perf_counter()

pca_model = SklearnPCA(n_components=k_95, svd_solver="randomized", random_state=0)
Z_train_95 = pca_model.fit_transform(X_train)
Z_test_95  = pca_model.transform(X_test)

print(f"[DEBUG] PCA model fit done in {t.perf_counter() - pca_model_start:.2f}s")
print(f"[DEBUG] Z_train_95 shape={Z_train_95.shape}, Z_test_95 shape={Z_test_95.shape}")

# Slice first k_80 components for the 80% variant (no second PCA fit so can run faster)
Z_train_80 = Z_train_95[:, :k_80]
Z_test_80  = Z_test_95[:, :k_80]
print(f"[DEBUG] Z_train_80 shape={Z_train_80.shape}, Z_test_80 shape={Z_test_80.shape}")

print(f"Raw: {X_train.shape} -> PCA95: {Z_train_95.shape} -> PCA80: {Z_train_80.shape}")

results = []

print("raw")
clf_raw, r = train(X_train, X_test, y_train, y_test, (16, 10), "2-layer Perceptron (raw)", max_iter=200)
results.append(r)

print("95")
clf_95, r = train(Z_train_95, Z_test_95, y_train, y_test, (16, 10), "2-layer Perceptron + PCA (95%)", max_iter=200)
results.append(r)

print("80")
clf_80, r = train(Z_train_80, Z_test_80, y_train, y_test, (16, 10), "2-layer Perceptron + PCA (80%)", max_iter=200)
results.append(r)

# Simple baseline table (raw vs PCA)
print("===== BASELINE TABLE =====")
print("Model | Input Dim | Accuracy | Macro F1 | Total(s)")
for r in results:
    print(f"{r['model']} | {r['input_dim']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['total_s']:.2f}")

print("[DEBUG] Finished all training/eval.")

