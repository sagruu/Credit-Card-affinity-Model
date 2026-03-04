import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# RFECV ausführen und speichern
def run_rfecv_on_processed(
    model,
    name,
    scoring_metric,
    X_train_processed,
    X_test_processed,
    y_train,
    feature_names,
    result_dict,
    model_dict,
    X_train_selected_dict,
    X_test_selected_dict,
    selected_feature_names_dict
):
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring_metric,
        n_jobs=-1
    )

    rfecv.fit(X_train_processed, y_train)
    model_dict[name] = rfecv

    selected = feature_names[rfecv.support_]
    print(f"{name}: {len(selected)} Features ausgewählt.")

    result_dict[name] = rfecv.cv_results_["mean_test_score"]
    X_train_selected_dict[name] = pd.DataFrame(X_train_processed, columns=feature_names)[selected]
    X_test_selected_dict[name] = pd.DataFrame(X_test_processed, columns=feature_names)[selected]
    selected_feature_names_dict[name] = selected

    # Speichern
    filename = f"{name}_rfecv_{scoring_metric}.pkl"
    filepath = os.path.join("models", "rfecv", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(rfecv, filepath)
    print(f"Modell gespeichert: {filepath}")


# -----------------------Plot-Funktion------------------------------------------------------

def plot_rfecv_results(rfecv_results_dict, scoring_name, rfecv_models_dict):
    plt.figure(figsize=(10, 6))

    for name, scores in rfecv_results_dict.items():
        line, = plt.plot(range(1, len(scores) + 1), scores, label=name, marker="o")
        color = line.get_color()
        n_selected = rfecv_models_dict[name].n_features_
        best_score = scores[n_selected - 1]

        plt.axvline(n_selected, color=color, linestyle='--', alpha=0.5)
        plt.plot(n_selected, best_score, marker="x", color="black", markersize=12)

        plt.text(
            n_selected + 3,
            best_score + 0.01,
            f"{name}: {n_selected}",
            fontsize=11,
            color="black",
            weight="bold",
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.9)
        )

    plt.xlabel("Anzahl ausgewählter Features")
    plt.ylabel(f"{scoring_name} (CV)")
    plt.title(f"RFECV-Vergleich der Modelle ({scoring_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# -----------------------------Features aus gespeicherter RFECV-Datei laden---------------------------

def load_rfecv_selected_features(model_name, metric, X_train_processed, X_test_processed, feature_names):
    path = f"models/rfecv/{model_name}_rfecv_{metric}.pkl"
    rfecv = joblib.load(path)
    mask = rfecv.support_
    selected = feature_names[mask]
    X_train_selected = pd.DataFrame(X_train_processed, columns=feature_names)[selected]
    X_test_selected = pd.DataFrame(X_test_processed, columns=feature_names)[selected]
    return X_train_selected, X_test_selected, selected
