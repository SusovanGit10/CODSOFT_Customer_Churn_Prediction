import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# Import preprocessing functions
from preprocessing import load_data, clean_data, encode_data, scale_data


def main():
    print("🚀 Starting training pipeline...\n")

    # ==============================
    # 1. Load & preprocess data
    # ==============================
    df = load_data("../data/Churn_Modelling.csv")
    df = clean_data(df)
    df, le_gender, le_geo = encode_data(df)

    print("✅ Data loaded and preprocessed\n")

    # ==============================
    # 2. Basic EDA
    # ==============================
    print("📊 Basic Data Insights:\n")

    print("Churn Distribution:\n", df['Exited'].value_counts())
    print("\nAverage Age by Churn:\n", df.groupby('Exited')['Age'].mean())
    print("\nGeography vs Churn:\n", df.groupby('Geography')['Exited'].mean())
    print("\nGender vs Churn:\n", df.groupby('Gender')['Exited'].mean())

    print("\n====================================\n")

    # ==============================
    # 3. Split features & target
    # ==============================
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("✅ Data split completed\n")

    # ==============================
    # 4. Handle imbalance (SMOTE)
    # ==============================
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("✅ Applied SMOTE (class balancing)\n")

    # ==============================
    # 5. Feature scaling
    # ==============================
    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("✅ Feature scaling applied\n")

    # ==============================
    # 6. Train Multiple Models
    # ==============================
    print("⏳ Training multiple models...\n")

    # Logistic Regression (improved)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')

    # Random Forest with proper scoring
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_params,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    # Gradient Boosting (FIXED)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf_grid,
        "Gradient Boosting": gb
    }

    results = []

    for name, model in models.items():
        print(f"🔹 Training {name}...")

        model.fit(X_train, y_train)

        # Extract best RF model
        if name == "Random Forest":
            model = model.best_estimator_

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results.append((name, model, acc, roc))

        print(f"{name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}\n")

    # ==============================
    # 7. Select Best Model (Improved)
    # ==============================
    best_name, best_model, best_acc, best_roc = sorted(
        results,
        key=lambda x: (x[3], x[2]),  # ROC-AUC first, then accuracy
        reverse=True
    )[0]

    print("🏆 Best Model Selected:")
    print(f"Model: {best_name}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"ROC-AUC: {best_roc:.4f}\n")

    # ==============================
    # 8. Final Evaluation (Threshold tuned)
    # ==============================
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Business-focused threshold
    y_pred = (y_prob > 0.4).astype(int)

    print("📊 Final Model Evaluation:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ==============================
    # 9. Feature Importance
    # ==============================
    if hasattr(best_model, "feature_importances_"):
        print("\n📌 Plotting Feature Importance...\n")

        importances = best_model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.xlabel("Importance")
        plt.title(f"{best_name} Feature Importance")
        plt.tight_layout()
        plt.show()

    # ==============================
    # 10. Save best model
    # ==============================
    joblib.dump(best_model, "../models/model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    joblib.dump(le_gender, "../models/le_gender.pkl")
    joblib.dump(le_geo, "../models/le_geo.pkl")

    print(f"\n💾 {best_name} saved as final model!")


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    main()