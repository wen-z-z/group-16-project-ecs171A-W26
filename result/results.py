import pandas as pd

# gathers the results from the MLP_model.py file and saves them to a csv file
# using the csv file, the results can be displayed in the web app
results = [
    {
        "model": "MLP (raw)",
        "input_dim": 784,
        "accuracy": acc_raw,
        "macro_f1": f1_raw,
        "runtime_sec": time_raw
    },
    {
        "model": "MLP + PCA (95%)",
        "input_dim": k95,
        "accuracy": acc_pca95,
        "macro_f1": f1_pca95,
        "runtime_sec": time_pca95
    }
]

df = pd.DataFrame(results)
df.to_csv("results/baseline_results.csv", index=False)