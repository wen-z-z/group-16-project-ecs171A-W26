import os
import numpy as np
import pandas as pd
import copy
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


def train(X_train, X_test, y_train, y_test, layer_size, model_name, max_iter=200, warm_start=False):
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
      tol=1e-4,
      warm_start=warm_start
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

# helper to evaluate a trained sklearn MLP quickly
def eval_model(clf, X_test, y_test):
  y_pred = clf.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average="macro")
  return acc, f1


# build a global mask over all weight matrices (same shapes as coefs_)
def init_masks_like_coefs(clf):
  return [np.ones_like(W, dtype=bool) for W in clf.coefs_]


# apply masks to coefs_ in-place (zero out pruned weights)
def apply_masks_inplace(clf, masks):
  for i in range(len(clf.coefs_)):
    clf.coefs_[i] = clf.coefs_[i] * masks[i]


# iterative global magnitude pruning with retraining
def iterative_magnitude_pruning(
    clf,
    X_train, y_train,
    X_test, y_test,
    prune_per_round=0.2,   # remove 20% of remaining weights each round
    rounds=5,
    retrain_max_iter=200,   # retrain fully each round (can change to 50 for faster runtime but less retraining)
    seed=0
):
  """
  Returns a list of dicts with pruning progress.
  Global magnitude pruning: each round, prune the smallest |weights| among all unpruned weights.
  """

  rng = np.random.default_rng(seed)

  # Create masks (True = keep, False = pruned)
  masks = init_masks_like_coefs(clf)

  # Record initial performance
  acc0, f10 = eval_model(clf, X_test, y_test)
  total_weights = int(sum(W.size for W in clf.coefs_))
  kept_weights = int(sum(m.sum() for m in masks))
  history = [{
      "round": 0,
      "kept_frac": kept_weights / total_weights,
      "pruned_frac": 1 - kept_weights / total_weights,
      "accuracy": acc0,
      "macro_f1": f10
  }]

  print(f"[DEBUG] Pruning start: total weights={total_weights}, kept={kept_weights}")

  # Reuse the same clf object
  clf.warm_start = True

  for r in range(1, rounds + 1):
    # Collect magnitudes of all currently-unpruned weights
    mags = []
    for W, m in zip(clf.coefs_, masks):
      mags.append(np.abs(W[m]))
    all_mags = np.concatenate(mags) if len(mags) else np.array([])

    if all_mags.size == 0:
      print("[DEBUG] No weights left to prune.")
      break

    # Determine how many more weights to prune this round (fraction of remaining)
    remaining = all_mags.size
    to_prune = int(np.floor(prune_per_round * remaining))
    to_prune = max(to_prune, 1)  # prune at least 1 weight per round

    # Threshold = value at the to_prune-th smallest magnitude
    # (partition is faster than sorting everything)
    thresh = np.partition(all_mags, to_prune - 1)[to_prune - 1]

    # Update masks: prune weights with |w| <= thresh (but only those currently unpruned)
    pruned_now = 0
    for i in range(len(clf.coefs_)):
      W = clf.coefs_[i]
      m = masks[i]
      prune_here = (np.abs(W) <= thresh) & m
      pruned_now += int(prune_here.sum())
      masks[i][prune_here] = False

    # Apply masks to zero out pruned weights
    apply_masks_inplace(clf, masks)

    # Retrain briefly (warm-start continues from current weights)
    print(f"[DEBUG] Prune round {r}: threshold={thresh:.6g}, pruned_now={pruned_now}, retraining...")
    clf.max_iter = retrain_max_iter
    clf.fit(X_train, y_train)

    # enforce masks again (optimizer may change zeros slightly)
    apply_masks_inplace(clf, masks)

    acc, f1 = eval_model(clf, X_test, y_test)
    kept_weights = int(sum(m.sum() for m in masks))
    history.append({
        "round": r,
        "kept_frac": kept_weights / total_weights,
        "pruned_frac": 1 - kept_weights / total_weights,
        "accuracy": acc,
        "macro_f1": f1
    })

    print(f"[DEBUG] Round {r} done: kept_frac={kept_weights/total_weights:.4f}, acc={acc:.4f}, macro_f1={f1:.4f}")

  return history


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
os.makedirs("results", exist_ok=True)
plt.savefig("results/explained_variance.png", dpi=150)
plt.close()

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

# Save results to CSV for the Streamlit app
os.makedirs("results", exist_ok=True)
df = pd.DataFrame([{
    "model": r["model"],
    "input_dim": r["input_dim"],
    "accuracy": r["accuracy"],
    "macro_f1": r["macro_f1"],
    "runtime_sec": r["total_s"]
} for r in results])
df.to_csv("results/baseline_results.csv", index=False)
print(f"[DEBUG] Saved results to results/baseline_results.csv")

print("[DEBUG] Finished all training/eval.")

# =========================
# Iterative Pruning Experiment (on one model)
# =========================
# Pick which representation to prune: raw, PCA95, or PCA80
PRUNE_ON = "RAW"  # "RAW", "PCA95", "PCA80"

if PRUNE_ON == "RAW":
  Xtr, Xte = X_train, X_test
  base_name = "Pruning on RAW"
elif PRUNE_ON == "PCA95":
  Xtr, Xte = Z_train_95, Z_test_95
  base_name = "Pruning on PCA95"
else:
  Xtr, Xte = Z_train_80, Z_test_80
  base_name = "Pruning on PCA80"

print(f"[DEBUG] Starting iterative magnitude pruning experiment: {base_name}")

# Train a fresh model to prune (so you don't mutate your baseline clf_*)
clf_prune, _ = train(Xtr, Xte, y_train, y_test, (16, 10), f"{base_name} (pre-prune)", max_iter=200)

prune_history = iterative_magnitude_pruning(
    clf_prune,
    Xtr, y_train,
    Xte, y_test,
    prune_per_round=0.2,   # prune 20% of remaining weights each round (in results I also ran with 0.1)
    rounds=6,
    retrain_max_iter=200
)

# Save pruning history to CSV
os.makedirs("results", exist_ok=True)
prune_df = pd.DataFrame(prune_history)
prune_csv = f"results/pruning_history_{PRUNE_ON.lower()}.csv"
prune_df.to_csv(prune_csv, index=False)
print(f"[DEBUG] Saved pruning history to {prune_csv}")

# Plot pruning trend
plt.figure(figsize=(8, 5))
plt.plot(prune_df["pruned_frac"], prune_df["accuracy"], marker="o")
plt.xlabel("Pruned Fraction (global magnitude)")
plt.ylabel("Test Accuracy")
plt.title(f"Iterative Magnitude Pruning Trend ({PRUNE_ON})")
plt.tight_layout()
plot_path = f"results/pruning_trend_{PRUNE_ON.lower()}.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[DEBUG] Saved pruning plot to {plot_path}")
