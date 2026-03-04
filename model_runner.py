import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def encode_categorical_features(df):
    df = df.copy()
    # One-hot encoding for 'gender' and 'region_client'
    df = pd.get_dummies(df, columns=["gender", "region_client"], drop_first=True)
    return df


def preprocess_features(df, skew_thresh=1.0, sparsity_thresh=0.95, exclude=[]):
    """
    Prepares numeric features for ML:
    - converts 'gender' and 'region_client' to one-hot features
    - detects skewed distributions -> log1p (if >= 0) or PowerTransformer (if negative values allowed)
    - detects sparse features -> binary or drop
    - scales all transformed features
    - all transformations overwrite the original columns
    """

    df = df.copy()

    # 0. Categorical Encoding
    df = encode_categorical_features(df)

    # 1. Filter numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude]

    # 2. Preparation
    log_transform_cols = []
    power_transform_cols = []
    binary_transform_cols = []
    drop_cols = []

    # 3. Analyze skewness + sparsity
    for col in numeric_cols:
        zero_ratio = (df[col] == 0).mean()
        skewness = df[col].skew()

        if zero_ratio > sparsity_thresh:
            if df[col].nunique() > 2:
                binary_transform_cols.append(col)
            else:
                drop_cols.append(col)
        elif abs(skewness) > skew_thresh:
            if (df[col] >= 0).all():
                log_transform_cols.append(col)
            else:
                power_transform_cols.append(col)

    # 4. Binary transformation (replaces original)
    for col in binary_transform_cols:
        df[col] = (df[col] > 0).astype(int)

    # 5. Log transformation (replaces original)
    for col in log_transform_cols:
        df[col] = np.log1p(df[col].clip(lower=0))

    # 6. Power transformation (replaces original)
    pt = PowerTransformer(method='yeo-johnson')
    for col in power_transform_cols:
        df[[col]] = pt.fit_transform(df[[col]])

    # 7. Drop unusable columns
    df.drop(columns=drop_cols, inplace=True)

    # 8. Scale all remaining numeric features
    scale_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # 9. Summary table
    summary = pd.DataFrame({
        "Feature": numeric_cols,
        "Zero Ratio": [round((df[c]==0).mean(), 3) if c in df.columns else np.nan for c in numeric_cols],
        "Skewness": [round(df[c].skew(), 3) if c in df.columns else np.nan for c in numeric_cols],
        "Action": [
            "drop" if c in drop_cols else
            "binary" if c in binary_transform_cols else
            "log+scale" if c in log_transform_cols else
            "power+scale" if c in power_transform_cols else
            "keep"
            for c in numeric_cols
        ]
    })

    return df, summary



class ModelRunner:
    def __init__(self, cv_folds=10):
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.results = []

    def fit(self, name, model, X_train, y_train, X_test, y_test, param_grid=None, reduced=False):
        print(f"\nTraining: {name}")

        # Prepare result file
        result_file = "results/results_reduced.pkl" if reduced else "results/results_all.pkl"
        os.makedirs("results", exist_ok=True)
        if os.path.exists(result_file):
            self.results = joblib.load(result_file)
        else:
            self.results = []

        # Skip if already trained
        if any(r["model"] == name for r in self.results):
            print(f"Model '{name}' has already been trained. Skipping.")
            return

        # Pipeline without preprocessor
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("classifier", model)
        ])

        scoring = {
            'precision': 'precision',
            'roc_auc': 'roc_auc'
        }

        if param_grid:
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=self.cv,
                scoring=scoring,
                refit='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe
            best_params = "Default"

        # Test predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # CV metrics
        cv_scores = cross_validate(
            best_model,
            X_train,
            y_train,
            cv=self.cv,
            scoring={
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score),
                "recall": make_scorer(recall_score),
                "f1": make_scorer(f1_score),
                "roc_auc": "roc_auc"
            },
            n_jobs=-1,
            return_train_score=False
        )

        # Test metrics
        test_scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        result_entry = {
            "model": name,
            "params": best_params,
            "test_metrics": test_scores,
            "cv_metrics": {
                metric: (np.mean(scores), np.std(scores))
                for metric, scores in cv_scores.items()
                if metric.startswith("test_")
            },
            "cv_raw": {
                metric.replace("test_", ""): scores.tolist()
                for metric, scores in cv_scores.items()
                if metric.startswith("test_")
            }
        }

        # Add result and save immediately
        self.results.append(result_entry)
        joblib.dump(self.results, result_file)
        print(f"Results saved to: {result_file}")

        # Save model
        folder = "models/models_reduced_features" if reduced else "models/models_all_features"
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{name}.pkl")
        joblib.dump(best_model, filepath)
        print(f"Model saved to: {filepath}")

        result_file = "results/results_reduced.pkl" if reduced else "results/results_all.pkl"
        os.makedirs("results", exist_ok=True)
        joblib.dump(self.results, result_file)
        print(f"Results saved to: {result_file}")

        # Save individual result separately
        result_single_file = os.path.join(folder, f"{name}_result.pkl")
        joblib.dump(result_entry, result_single_file)
        print(f"Individual result saved to: {result_single_file}")

    def get_results(self, include_cv_raw=False):
        data = []
        for r in self.results:
            row = {
                "Model": r["model"],
                "Best Params": r["params"],
                **{f"Test {k}": v for k, v in r["test_metrics"].items()},
                **{f"CV {k[5:]} (mean)": round(v[0], 3) for k, v in r["cv_metrics"].items()},
                **{f"CV {k[5:]} (std)": round(v[1], 3) for k, v in r["cv_metrics"].items()},
            }
            if include_cv_raw:
                row["cv_raw"] = r["cv_raw"]
            data.append(row)
        return pd.DataFrame(data)

    def load_results(self, reduced=False):
        result_file = "results/results_reduced.pkl" if reduced else "results/results_all.pkl"
        if os.path.exists(result_file):
            self.results = joblib.load(result_file)
            print(f"Results loaded from {result_file}.")
        else:
            print(f"No saved results found at: {result_file}")


