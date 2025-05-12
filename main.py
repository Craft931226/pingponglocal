import argparse, pandas as pd, numpy as np, lightgbm as lgb
from pathlib import Path
from feature_utils import generate_features, aggregate_group_prob
from model_utils import TARGETS, build_scaler, build_model
from model_utils import cv_evaluate, save_model, load_model, tune_lgb_params
from train_val_utils import train_validate_split, evaluate_model, evaluate_validation_set
import warnings
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Log the start of the script
logging.info("Script started.")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

def load_features(feat_dir):
    X, uid_idx = [], []
    for p in Path(feat_dir).glob("*.csv"):
        df = pd.read_csv(p)
        X.append(df.values)         # (n_swing, feat_dim)
        uid_idx.extend([int(p.stem)] * len(df))
    return np.vstack(X), np.array(uid_idx)

def prepare_train():
    # 1. 產生特徵
    # generate_features("./train_data", "train_info.csv", "tabular_data_train")

    # 2. 讀取 info & 特徵
    info = pd.read_csv("train_info.csv")
    X, uid_idx = load_features("tabular_data_train")
    groups = info.set_index("unique_id").loc[uid_idx, "player_id"].values

    # 3. 數據標準化
    scaler = build_scaler(X)
    X_scaled = scaler.transform(X)
    
    # 儲存訓練/驗證集的分割
    all_targets = {}
    holdout_data = None
    
    # 4. 對每個 target 建模
    for col, meta in TARGETS.items():
        print(f"\nTraining for {col}:")
        logging.info(f"\nTraining for {col}:")
        y = info.set_index("unique_id").loc[uid_idx, col].values
        
        # 拆分訓練集和驗證集
        data_dict = train_validate_split(X_scaled, y, groups)
            # 👉 使用 Optuna 找最佳超參數
        best_params = tune_lgb_params(
            data_dict["X_train"], data_dict["y_train"], data_dict["groups_train"], meta, n_trials=20
        )
        print(f"Best params for {col}: {best_params}")
        logging.info(f"Best params for {col}: {best_params}")
        # 👉 用最佳參數建模型
        mdl = build_model(y, meta, params_override=best_params)

        if holdout_data is None:
            holdout_data = {
                'X_val': data_dict['X_val'],
                'X_train': data_dict['X_train'],
                'y_val': {},
                'y_train': {},
                'groups_val': data_dict['groups_val']
            }
        
        holdout_data['y_val'][col] = data_dict['y_val']
        holdout_data['y_train'][col] = data_dict['y_train']
        
        # 訓練和驗證
        # mdl = build_model(y, meta)
        val_score, trained_model = evaluate_model(mdl, data_dict, meta)
        print(f"Training Score for {col}: {val_score:.4f}")
        logging.info(f"Training Score for {col}: {val_score:.4f}")
        save_model(trained_model, scaler, col)
        all_targets[col] = {'model': trained_model, 'scaler': scaler}
        

    print("\nEvaluating validation set:")
    scores, avg_score = evaluate_validation_set(holdout_data, all_targets, TARGETS)
    np.save("split_uid_val.npy", 
            np.unique(uid_idx[np.isin(groups, holdout_data['groups_val'])]))
    print("\n✅ Models saved to ./models/")

def predict_test():
    # 1. 產生特徵
    generate_features("./test_data", "test_info.csv", "tabular_data_test")

    # 2. 預先加載所有模型
    models = {col: load_model(col) for col in TARGETS.keys()}
    
    # 3. 批量讀取所有測試數據
    test_files = sorted(Path("tabular_data_test").glob("*.csv"))
    all_uids = []
    all_features = []
    
    for p in test_files:
        uid = int(p.stem)
        df = pd.read_csv(p)
        all_uids.append(uid)
        all_features.append(df.values)
    
    sub_rows = []
    for idx, uid in enumerate(all_uids):
        X = all_features[idx]
        row = {"unique_id": uid}

        for col, meta in TARGETS.items():
            bundle = models[col]
            scaler = bundle["scaler"]
            model = bundle["model"]

            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            grp = aggregate_group_prob(proba)[0]

            if meta["type"] == "bin":
                pos_idx = np.where(model.classes_ == 1)[0][0]
                row[col] = grp[pos_idx]
                continue

            needed_labels = [0, 1, 2] if col == "play years" else [2, 3, 4, 5]
            for lbl in needed_labels:
                row[f"{col}_{lbl}"] = 0.0

            for idx, lbl in enumerate(model.classes_):
                row[f"{col}_{lbl}"] = grp[idx]

        sub_rows.append(row)

    sub_cols = ["unique_id", "gender", "hold racket handed",
                "play years_0","play years_1","play years_2",
                "level_2","level_3","level_4","level_5"]
    df_temp = pd.DataFrame(sub_rows)
    df_temp = df_temp.reindex(columns=sub_cols, fill_value=0.0)
    submission = df_temp[sub_cols]
    # 使用 DataFrame 批量處理
    submission.to_csv("submission.csv", index=False, float_format="%.8f")
    print("✅  submission.csv ready!")

if __name__ == "__main__": 
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train","predict"])
    args = ap.parse_args()

    try:
        if args.mode == "train":
            prepare_train()
        else:
            predict_test()
        logging.info("Script finished successfully.")
    except Exception as e:
        logging.error(f"Script encountered an error: {e}")
        raise
