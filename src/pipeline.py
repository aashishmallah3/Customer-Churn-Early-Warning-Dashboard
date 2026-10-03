"""
Telco Customer Churn Pipeline:
- Data Ingestion & Preprocessing
- Logistic Regression Baseline
- Tuned XGBoost with scale_pos_weight
- Financial Cost-Benefit Optimization
- SHAP Feature Attribution & Top Drivers Extraction
- Artifact Serialization for Streamlit Dashboard
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb
import shap

def load_and_clean_data(csv_path: str = "data/Telco-Customer-Churn.csv") -> pd.DataFrame:
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean TotalCharges: convert empty whitespace to 0.0 (new customers with tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    
    # Clean target
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    print(f"Loaded {len(df)} records. Churn rate: {df['Churn_Binary'].mean() * 100:.2f}%")
    return df

def run_pipeline():
    df = load_and_clean_data()
    
    # Feature specifications
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    
    X = df[numeric_features + categorical_features]
    y = df["Churn_Binary"]
    customer_ids = df["customerID"]
    
    # Stratified Train/Test Split (80/20)
    print("Splitting data into train and test sets (stratified 80/20)...")
    X_train, X_test, y_train, y_test, ids_train, ids_test, orig_train, orig_test = train_test_split(
        X, y, customer_ids, df, test_size=0.20, random_state=42, stratify=y
    )
    
    # Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(cat_feature_names)
    print(f"Total processed features: {len(feature_names)}")
    
    # ---------------------------------------------------------
    # 1. Baseline Model: Logistic Regression
    # ---------------------------------------------------------
    print("\n--- Training Baseline Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_proc, y_train)
    
    y_test_pred_lr = log_reg.predict(X_test_proc)
    y_test_prob_lr = log_reg.predict_proba(X_test_proc)[:, 1]
    
    lr_roc_auc = roc_auc_score(y_test, y_test_prob_lr)
    lr_pr_auc = average_precision_score(y_test, y_test_prob_lr)
    lr_acc = accuracy_score(y_test, y_test_pred_lr)
    lr_prec = precision_score(y_test, y_test_pred_lr)
    lr_rec = recall_score(y_test, y_test_pred_lr)
    lr_f1 = f1_score(y_test, y_test_pred_lr)
    
    print(f"Logistic Regression Results:")
    print(f"  ROC-AUC:  {lr_roc_auc:.4f}")
    print(f"  PR-AUC:   {lr_pr_auc:.4f}")
    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  Precision:{lr_prec:.4f}")
    print(f"  Recall:   {lr_rec:.4f}")
    print(f"  F1-Score: {lr_f1:.4f}")
    
    # ---------------------------------------------------------
    # 2. Tuned XGBoost Model with scale_pos_weight
    # ---------------------------------------------------------
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"\n--- Training XGBoost (scale_pos_weight = {scale_pos_weight:.3f}) ---")
    
    xgb_base = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.08, 0.15],
        "n_estimators": [100, 150],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_train_proc, y_train)
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    y_test_pred_xgb = best_xgb.predict(X_test_proc)
    y_test_prob_xgb = best_xgb.predict_proba(X_test_proc)[:, 1]
    
    xgb_roc_auc = roc_auc_score(y_test, y_test_prob_xgb)
    xgb_pr_auc = average_precision_score(y_test, y_test_prob_xgb)
    xgb_acc = accuracy_score(y_test, y_test_pred_xgb)
    xgb_prec = precision_score(y_test, y_test_pred_xgb)
    xgb_rec = recall_score(y_test, y_test_pred_xgb)
    xgb_f1 = f1_score(y_test, y_test_pred_xgb)
    
    print(f"XGBoost Test Results:")
    print(f"  ROC-AUC:  {xgb_roc_auc:.4f}")
    print(f"  PR-AUC:   {xgb_pr_auc:.4f}")
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(f"  Precision:{xgb_prec:.4f}")
    print(f"  Recall:   {xgb_rec:.4f}")
    print(f"  F1-Score: {xgb_f1:.4f}")
    
    # ---------------------------------------------------------
    # 3. Cost-Based Decision Threshold Optimization
    # ---------------------------------------------------------
    print("\n--- Computing Cost-Optimal Decision Threshold ---")
    cost_offer = 20.0     # Cost of a retention incentive offer
    cost_churn = 500.0    # Value/Cost of losing a customer
    
    thresholds = np.linspace(0.05, 0.95, 181)
    cost_records = []
    
    # Baseline: Do nothing (never offer retention, all churners leave)
    total_actual_churners = np.sum(y_test == 1)
    cost_do_nothing = total_actual_churners * cost_churn
    
    for t in thresholds:
        pred_t = (y_test_prob_xgb >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_t).ravel()
        
        # Total cost: Unnecessary offers (FP * $20) + Lost Customers (FN * $500) + Interventions on actual churners (TP * $20)
        total_cost = (fn * cost_churn) + ((tp + fp) * cost_offer)
        net_savings = cost_do_nothing - total_cost
        
        cost_records.append({
            "threshold": float(t),
            "total_cost": float(total_cost),
            "net_savings": float(net_savings),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": float(precision_score(y_test, pred_t, zero_division=0)),
            "recall": float(recall_score(y_test, pred_t, zero_division=0)),
            "f1": float(f1_score(y_test, pred_t, zero_division=0)),
        })
        
    cost_df = pd.DataFrame(cost_records)
    optimal_idx = cost_df["total_cost"].idxmin()
    optimal_threshold = cost_df.loc[optimal_idx, "threshold"]
    min_cost = cost_df.loc[optimal_idx, "total_cost"]
    max_savings = cost_df.loc[optimal_idx, "net_savings"]
    
    # Cost at default 0.50
    default_50_row = cost_df.iloc[(cost_df["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
    
    print(f"Cost of Doing Nothing: ${cost_do_nothing:,.2f}")
    print(f"Cost at default 0.50 threshold: ${default_50_row['total_cost']:,.2f} (Savings: ${default_50_row['net_savings']:,.2f})")
    print(f"Cost-Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Minimum Cost at Optimal: ${min_cost:,.2f}")
    print(f"Maximum Net Savings at Optimal: ${max_savings:,.2f}")
    print(f"Incremental Gain over 0.50 cutoff: ${default_50_row['total_cost'] - min_cost:,.2f}")
    
    # ---------------------------------------------------------
    # 4. SHAP Feature Attribution
    # ---------------------------------------------------------
    print("\n--- Computing SHAP Values (TreeExplainer) ---")
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test_proc)
    expected_value = explainer.expected_value
    
    # Compute Top 3 Churn Drivers for every customer in test set
    print("Extracting top 3 individual drivers per customer...")
    top_drivers_list = []
    
    for i in range(len(X_test_proc)):
        row_shap = shap_values[i]
        # Sort features by positive impact on churn risk (highest positive SHAP value first)
        pos_indices = np.argsort(row_shap)[::-1]
        
        drivers = []
        for idx in pos_indices[:3]:
            feat_name = feature_names[idx]
            val_proc = X_test_proc[i, idx]
            shap_impact = float(row_shap[idx])
            
            # Format friendly name
            clean_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", " ")
            drivers.append({
                "feature": clean_name,
                "raw_feature": feat_name,
                "shap_value": shap_impact,
                "feature_index": int(idx),
            })
        top_drivers_list.append(drivers)
        
    # Build enriched test dataframe
    test_results_df = orig_test.copy()
    test_results_df["test_idx"] = np.arange(len(X_test_proc))
    test_results_df["Churn_Prob"] = y_test_prob_xgb
    test_results_df["Churn_Pred_Default"] = (y_test_prob_xgb >= 0.50).astype(int)
    test_results_df["Churn_Pred_Optimal"] = (y_test_prob_xgb >= optimal_threshold).astype(int)
    test_results_df["Top_Driver_1"] = [d[0]["feature"] for d in top_drivers_list]
    test_results_df["Top_Driver_1_Impact"] = [d[0]["shap_value"] for d in top_drivers_list]
    test_results_df["Top_Driver_2"] = [d[1]["feature"] for d in top_drivers_list]
    test_results_df["Top_Driver_2_Impact"] = [d[1]["shap_value"] for d in top_drivers_list]
    test_results_df["Top_Driver_3"] = [d[2]["feature"] for d in top_drivers_list]
    test_results_df["Top_Driver_3_Impact"] = [d[2]["shap_value"] for d in top_drivers_list]
    
    # Assign Risk Tiers
    def get_risk_tier(p):
        if p >= 0.70: return "Critical (>70%)"
        if p >= 0.40: return "High (40-70%)"
        if p >= 0.20: return "Medium (20-40%)"
        return "Low (<20%)"
        
    test_results_df["Risk_Tier"] = test_results_df["Churn_Prob"].apply(get_risk_tier)
    test_results_df = test_results_df.sort_values(by="Churn_Prob", ascending=False).reset_index(drop=True)
    
    # Global feature importance summary
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance_df = pd.DataFrame({
        "feature": [f.replace("cat__", "").replace("num__", "") for f in feature_names],
        "importance": mean_abs_shap,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 5. Serialization of Artifacts
    # ---------------------------------------------------------
    print("\n--- Serializing Models & Metrics to artifacts/ ---")
    artifacts = {
        "preprocessor": preprocessor,
        "baseline_model": log_reg,
        "xgb_model": best_xgb,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics_summary": {
            "logistic_regression": {
                "roc_auc": lr_roc_auc,
                "pr_auc": lr_pr_auc,
                "accuracy": lr_acc,
                "precision": lr_prec,
                "recall": lr_rec,
                "f1": lr_f1,
            },
            "xgboost": {
                "roc_auc": xgb_roc_auc,
                "pr_auc": xgb_pr_auc,
                "accuracy": xgb_acc,
                "precision": xgb_prec,
                "recall": xgb_rec,
                "f1": xgb_f1,
            },
        },
        "cost_analysis": {
            "cost_offer": cost_offer,
            "cost_churn": cost_churn,
            "cost_do_nothing": cost_do_nothing,
            "optimal_threshold": optimal_threshold,
            "min_cost": min_cost,
            "max_savings": max_savings,
            "default_50_row": default_50_row.to_dict(),
            "cost_df": cost_df,
        },
        "shap_data": {
            "expected_value": float(expected_value) if not hasattr(expected_value, "__len__") else float(expected_value[0]),
            "shap_values": shap_values,
            "X_test_proc": X_test_proc,
            "global_importance": global_importance_df,
            "top_drivers_list": top_drivers_list,
        },
        "test_results_df": test_results_df,
        "full_df": df,
    }
    
    out_path = "artifacts/model_artifacts.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Artifacts successfully saved to {out_path} ({os.path.getsize(out_path) / (1024*1024):.2f} MB)")
    print("Pipeline run completed successfully!")

if __name__ == "__main__":
    run_pipeline()
