# Utilities

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from IPython.display import display
import joblib
import os
import json
import re
import pkg_resources


# Function to update requirements.txt (already existing packages are not overwritten - first version number is kept)
# Requirements should still be manually verified at the end !!!!!

def save_explicit_imports_from_notebook(notebook_path, output_path="requirements.txt"):
    """
    Extracts imported packages from a Jupyter notebook and saves them to a requirements.txt file.
    Existing packages in the file are not overwritten; only new packages are added.
    :param notebook_path: Path to the Jupyter notebook
    :param output_path: Path to the requirements.txt file (default: "requirements.txt")
    """
    # Read existing requirements.txt if present
    existing_requirements = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "==" in line:
                    pkg, version = line.split("==", 1)
                    existing_requirements[pkg.lower()] = f"{pkg}=={version}"

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    imported_packages = set()
    import_re = re.compile(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)')

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                match = import_re.match(line)
                if match:
                    root_package = match.group(1).split('.')[0]
                    imported_packages.add(root_package)

    # Retrieve version information
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    new_requirements = []

    for pkg in sorted(imported_packages):
        pkg_lower = pkg.lower()
        if pkg_lower in existing_requirements:
            continue  # do not overwrite
        if pkg_lower in installed:
            version = installed[pkg_lower]
            new_requirements.append(f"{pkg}=={version}")
        else:
            print(f"Warning: {pkg} not found in the current environment.")

    # Update requirements.txt
    with open(output_path, "a", encoding="utf-8") as f:
        if new_requirements:
            f.write("\n" + "\n".join(new_requirements))
            print(f"{len(new_requirements)} new packages added.")
        else:
            print("No new packages found - requirements.txt unchanged.")

# Usage
#save_explicit_imports_from_notebook("main.ipynb")


# Helper Functions
def analyze_table(df, n_head = 3):
    # % missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    zero_values = (df == 0).sum()
    # unique values
    unique_values = df.nunique()
    data_types = df.dtypes

    analysis = pd.DataFrame({
        'Missing Percentage': missing_percentage,
        'Zero Values': zero_values,
        'Unique Values': unique_values,
        'Data Type': data_types
    })

    dimensions = f"{df.shape[0]} rows, {df.shape[1]} columns"
    print(f"Dimensions: {dimensions}")
    print("Head:")
    if n_head != 0:
        display(df.head(n_head))
    print("Analysis:")
    display(analysis)



def plot_distributions(df, cols=2):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns


    # Numeric columns
    if len(numeric_cols) > 0:
        rows = (len(numeric_cols) + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 3 * rows))
        for i, col in enumerate(numeric_cols):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(df[col], kde=True, bins=30, color="steelblue")
            plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

    # Categorical columns
    if len(categorical_cols) > 0:
        rows = (len(categorical_cols) + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 3.5 * rows))
        for i, col in enumerate(categorical_cols):
            plt.subplot(rows, cols, i + 1)
            df[col].value_counts().plot(kind="bar", color="orange")
            plt.title(f"Category Distribution of {col}")
            plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3, hspace=0.5)
        plt.show()

    # Date columns
    if len(datetime_cols) > 0:
        rows = (len(datetime_cols) + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 3 * rows))
        for i, col in enumerate(datetime_cols):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(df[col], bins=30, kde=False, color="green")
            plt.title(f"Date Distribution of {col}")
        plt.tight_layout()
        plt.show()


def plot_rollup(data, row, size=(6, 2)):
    """Plot the balance changes for the sample accounts with the event date. The green horizontal line represents 40'000 balance for comparison. The line is solid when there is a credit card and dotted when there is no credit card.
    Args:
    data: DataFrame with balance changes and balance by month
    cc_date: Date of credit card issuance
    """
    data = data.copy().reset_index()
    rollup_start = row["card_issued"] - pd.DateOffset(months=13)
    rollup_end = row["card_issued"] - pd.DateOffset(months=1)    
    data = data[data["date"] <= row["card_issued"] + pd.DateOffset(months=2)]
    data = data[data["date"] >= rollup_start]
    rollup_data = data.copy()
    rollup_data = rollup_data[
        (rollup_data["date"] >= rollup_start)
        & (rollup_data["date"] <= rollup_end)
    ]
    average_balance = rollup_data["balance"].mean()
    plt.figure(figsize=size)
    data.plot("date", "balance", marker="o")
    plt.title(f"Balance Over Time for Account: {row['account_id']}")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.ylim(-10000, data["balance"].max() * 1.1)
    plt.axvline(x=rollup_start, color="gray", alpha=0.5)  ## rollup start
    if row["has_card"]:
        plt.axvline(x=row["card_issued"], color="red")  ## credit card
    else:
        plt.axvline(
            x=row["card_issued"], color="orange", linestyle="dotted"
        )  ## no credit card
    plt.axhline(y=0, color="black")  ## 0 balance
    plt.axhline(y=50000, color="y", linestyle="--")  ## 50'000 balance
    plt.axhline(y=average_balance, color="green", linestyle="--")  ## average balance
    plt.show()


def plot_account_overview(data, account_id):
    df = data[data["account_id"] == account_id]

    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["balance"], marker="o", label="Balance")
    #plt.plot(df["date"], df["credit"], marker="^", label="Income")
    #plt.plot(df["date"], df["withdrawal"], marker="v", label="Expenses")
    plt.plot(df["date"], df["amount"], linestyle="--", label="Turnover (total)")

    plt.title(f"Balance and Turnover Over Time - Account {account_id}")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# Distribution Plots num.
def plot_feature_distributions(df, exclude=[]):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    n_cols = 3
    n_plots = len(numeric_cols)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Disable empty axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def save_model(model, name, folder="models_all_features"):
    path = os.path.join("models", folder, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(name, folder="models_all_features"):
    path = os.path.join("models", folder, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"Model loaded: {path}")
    return joblib.load(path)
