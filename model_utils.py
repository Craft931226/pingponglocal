import logging
import joblib, lightgbm as lgb
from pathlib import Path
import numpy as np, pandas as pd
import optuna
from optuna.trial import TrialState
# from sklearn.calibration import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TARGETS = {
    "gender":             {"type": "bin", "name": "gender"},
    "hold racket handed": {"type": "bin", "name": "hold"},
    "play years":         {"type": "multi", "num_class": 3, "name": "play_years"},
    "level":              {"type": "multi", "num_class": 4, "name": "level"},
}

def build_scaler(X: np.ndarray):
    sc = MinMaxScaler()
    return sc.fit(X)

def build_model(y, target_info, params_override: dict = None):
    if target_info["type"] == "bin":
        params = dict(
            objective   = "binary" if target_info["type"]=="bin" else "multiclass",
            metric      = "auc"    if target_info["type"]=="bin" else "multi_logloss",
            num_class   = target_info.get("num_class", None),
            learning_rate = 0.03,             # ↓ 降低學習率
            num_leaves    = 31,               # ↓ 減少複雜度
            n_estimators  = 1500,             # ↑ 配合較小 LR
            max_depth     = -1,
            min_child_samples = 20,
            bagging_freq      = 5,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.5,
            reg_lambda        = 0.5,
            bagging_fraction  = 0.9,
            feature_fraction  = 0.9,
            random_state      = 42,
            n_jobs            = -1,
            class_weight      = "balanced",
            boosting_type     = "dart",
            # boosting_type     = "goss",
            drop_rate         = 0.1,
            # learning_rate_drop  = 0.02,
            # num_iterations_drop = 2500,
        )

    else:
        params = dict(
            objective   = "binary" if target_info["type"]=="bin" else "multiclass",
            metric      = "auc"    if target_info["type"]=="bin" else "multi_logloss",
            num_class   = target_info.get("num_class", None),
            learning_rate = 0.01,             # ↓ 降低學習率
            num_leaves    = 31,               # ↓ 減少複雜度
            n_estimators  = 1500,             # ↑ 配合較小 LR
            max_depth     = -1,
            min_child_samples = 20,
            colsample_bytree  = 0.8,
            bagging_freq      = 5,
            reg_alpha         = 0.5,
            reg_lambda        = 0.5,
            bagging_fraction  = 0.9,
            feature_fraction  = 0.9,
            random_state      = 42,
            n_jobs            = -1,
            class_weight      = "balanced",
            boosting_type     = "dart",
            # boosting_type     = "goss",
            drop_rate         = 0.1,
        )
    # 如果有傳入最佳參數，就覆寫預設參數
    if params_override:
        params.update(params_override)
    return lgb.LGBMClassifier(**params)

def cv_evaluate(model, X, y, groups, target_info, early_stopping_rounds=30):
    gkf = GroupKFold(n_splits=5)
    scores = []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        eval_set = [(X_val, y_val)]
        
        # Use different metrics for binary and multiclass
        eval_metric = 'auc' if target_info["type"] == "bin" else 'multi_logloss'
        
        model.fit(X_tr, y_tr,
                 eval_set=eval_set,
                 eval_metric=eval_metric,
                 callbacks=[lgb.early_stopping(early_stopping_rounds)])
                 
        proba = model.predict_proba(X_val)

        # ---------- Binary ----------
        if target_info["type"] == "bin":
            pos_prob = proba[:, 1] if proba.ndim == 2 else proba.ravel()
            scores.append(roc_auc_score(y_val, pos_prob))
            continue

        # ---------- Multi-class ----------
        present = np.unique(y_val)             
        if len(present) == 1:                       
            continue                                

        col_of = {c: i for i, c in enumerate(model.classes_)}
        
        if len(present) == 2:
            pos_cls   = present[1]
            pos_prob  = proba[:, col_of[pos_cls]]
            y_bin = (y_val == pos_cls).astype(int)   
            score = roc_auc_score(y_bin, pos_prob)
            scores.append(score)
            continue

        proba_use = proba[:, [col_of[c] for c in present]]

        if proba_use.ndim > 1:
            proba_use = proba_use / proba_use.sum(axis=1, keepdims=True)

        score = roc_auc_score(
            y_val, proba_use,
            labels=present, average="micro", multi_class="ovr"
        )
        scores.append(score)

    return np.mean(scores) if scores else 0.5

def save_model(model, scaler, name):
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler},
                f"models/{name}.pkl")

def load_model(name):
    return joblib.load(f"models/{name}.pkl")

def tune_lgb_params(X_train, y_train, groups, target_info, n_trials: int = 30):
    """使用 Optuna 做 GroupKFold 的 LGB 超參數優化，回傳 best_params"""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    def objective(trial):
        param = {
            "objective": "binary" if target_info["type"]=="bin" else "multiclass",
            "metric":    "auc"    if target_info["type"]=="bin" else "multi_logloss",
            "learning_rate": trial.suggest_loguniform("lr", 1e-4, 1e-1),
            "num_leaves":    trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_loguniform("reg_alpha", 1e-3, 10),
            "reg_lambda":        trial.suggest_loguniform("reg_lambda", 1e-3, 10),
            "n_estimators": 1000,
            "random_state": 42,
            "n_jobs":       -1,
        }
        cv = GroupKFold(n_splits=3)
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train, groups):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if target_info["type"] == "multi" and len(np.unique(y_tr)) < target_info["num_class"]:
                return 0.0
            try:
                clf = lgb.LGBMClassifier(**param)
                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric=param["metric"],
                    callbacks=[lgb.early_stopping(30)]
                )
                proba = clf.predict_proba(X_val)
            except Exception:
                # 包含 unseen labels 或其他 fit 錯誤都視為此 trial 失敗
                return 0.0
            try:
                if target_info["type"] == "bin":
                    score = roc_auc_score(y_val, proba[:, 1])
                else:
                    import pandas as pd
                    y_ohe = pd.get_dummies(y_val)
                    score = roc_auc_score(
                        y_ohe, proba,
                        multi_class="ovr", average="micro"
                    )
                if np.isnan(score):
                    return 0.5
                scores.append(score)
            except ValueError:
                # Handle the case where all labels are the same
                return 0.5
        return sum(scores) / len(scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    if any(t.state == TrialState.COMPLETE for t in study.trials):
        return study.best_trial.params
    else:
        return {}