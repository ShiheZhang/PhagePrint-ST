import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import os
import joblib
import json
import sys
from datetime import datetime

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=UserWarning)

RESULTS_DIR = "random_forest_github_results"
RAW_DATA_DIR = os.path.join(RESULTS_DIR, "analysis_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def save_environment_info():
    env_info = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "script_path": os.path.abspath(__file__)
    }
    try:
        import importlib.metadata as metadata
        packages = ['pandas', 'numpy', 'scikit-learn', 'joblib']
        for pkg in packages:
            try:
                version = metadata.version(pkg)
                env_info[pkg] = version
            except:
                env_info[pkg] = "Not available"
    except:
        pass

    with open(f'{RESULTS_DIR}/environment_info.json', 'w') as f:
        json.dump(env_info, f, indent=2)


def prepare_data_for_rf(df):
    if 'Accession' in df.columns:
        X = df.drop(columns=['Accession', 'ST'])
    else:
        X = df.drop(columns=['ST'])

    y = df['ST']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    feature_names = X.columns.tolist()
    st_types_in_model = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, feature_names, st_types_in_model, le


def optimize_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True],
        'oob_score': [True]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=1)

    random_search = RandomizedSearchCV(
        rf, param_grid, n_iter=30, cv=3, scoring='accuracy',
        n_jobs=-1, random_state=42, verbose=1
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def calculate_and_save_metrics(rf_model, importance_df, X_test, y_test, le):
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    class_labels = le.classes_
    n_classes = len(class_labels)
    results = {}

    importance_df.to_csv(
        f'{RAW_DATA_DIR}/feature_importance_complete.csv', index=False)
    importance_df.head(30).to_csv(
        f'{RAW_DATA_DIR}/feature_importance_top30.csv', index=False)

    cumulative_importance = importance_df['cumulative_importance'].values
    try:
        idx_90 = np.argmax(cumulative_importance >= 0.9) + 1
    except:
        idx_90 = 0
    results['idx_90'] = idx_90

    pd.DataFrame({
        'n_features': range(1, len(importance_df) + 1),
        'cumulative_importance': cumulative_importance
    }).to_csv(f'{RAW_DATA_DIR}/cumulative_importance.csv', index=False)

    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(
        cm, dtype=float), where=(cm_sum != 0))

    pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(
        f'{RAW_DATA_DIR}/confusion_matrix_raw.csv')
    pd.DataFrame(cm_normalized, index=class_labels, columns=class_labels).to_csv(
        f'{RAW_DATA_DIR}/confusion_matrix_normalized.csv')

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
            f'{RAW_DATA_DIR}/roc_curve_binary.csv', index=False)
        results['roc_auc'] = roc_auc
    else:
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        fpr_micro, tpr_micro, _ = roc_curve(
            y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        pd.DataFrame({'fpr': fpr_micro, 'tpr': tpr_micro}).to_csv(
            f'{RAW_DATA_DIR}/roc_curve_micro.csv', index=False)

        results['roc_auc_micro'] = roc_auc_micro
        try:
            results['roc_auc_macro'] = roc_auc_score(
                y_test, y_pred_proba, multi_class='ovr', average='macro')
        except:
            results['roc_auc_macro'] = None

    return results


def calculate_diagnostics_data(rf_model, X_train, y_train, X_full=None, y_full=None):
    if hasattr(rf_model, 'oob_score_'):
        pass

    if X_full is not None and y_full is not None:
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                rf_model, X_full, y_full, cv=3, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring='accuracy', shuffle=True, random_state=42
            )

            lc_df = pd.DataFrame({
                'train_sizes': train_sizes,
                'train_scores_mean': train_scores.mean(axis=1),
                'train_scores_std': train_scores.std(axis=1),
                'test_scores_mean': test_scores.mean(axis=1),
                'test_scores_std': test_scores.std(axis=1)
            })
            lc_df.to_csv(
                f'{RAW_DATA_DIR}/learning_curve_data.csv', index=False)
        except Exception:
            pass


def save_final_artifacts(rf_model, importance_df, best_params, X_test, y_test, le, metrics_results, cv_scores, original_feature_names):
    y_pred = rf_model.predict(X_test)

    joblib.dump(rf_model, f'{RESULTS_DIR}/optimized_random_forest_model.pkl')
    joblib.dump(le, f'{RESULTS_DIR}/label_encoder.pkl')

    with open(f'{RESULTS_DIR}/feature_names.json', 'w') as f:
        json.dump(original_feature_names, f, indent=2)

    performance_stats = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'oob_accuracy': rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') else None,
        'cv_accuracy_mean': np.mean(cv_scores),
        'best_params': best_params
    }
    performance_stats.update(metrics_results)

    with open(f'{RESULTS_DIR}/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("RANDOM FOREST ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write(f"Test Accuracy: {performance_stats['test_accuracy']:.4f}\n")
        if performance_stats['oob_accuracy']:
            f.write(f"OOB Accuracy: {performance_stats['oob_accuracy']:.4f}\n")
        f.write(
            f"CV Accuracy (Mean): {performance_stats['cv_accuracy_mean']:.4f}\n")
        f.write(f"\nBest Parameters:\n{json.dumps(best_params, indent=2)}\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))


def main_analysis(df):
    save_environment_info()

    data = prepare_data_for_rf(df)
    X_train, X_test, y_train, y_test, feature_names, st_types, le = data

    X_full = pd.concat([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    model, best_params = optimize_random_forest(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    metrics = calculate_and_save_metrics(
        model, importance_df, X_test, y_test, le)

    calculate_diagnostics_data(model, X_train, y_train, X_full, y_full)

    save_final_artifacts(model, importance_df, best_params,
                         X_test, y_test, le, metrics, cv_scores, feature_names)


if __name__ == "__main__":

    data_path = 'vOTU_01_matrix_top20.csv'

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    elif 'DATA_PATH' in os.environ:
        data_path = os.environ['DATA_PATH']
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0)
        main_analysis(df)
    else:
        print(f"File not found: {data_path}")
        print("Please provide a valid CSV file path as command line argument or set DATA_PATH environment variable.")
