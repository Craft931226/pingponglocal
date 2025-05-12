import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
from sklearn.linear_model import LogisticRegression

def stacking_predict(models_dict, X_val, meta_list):
    """取所有基模型機率當特徵，訓練簡單 LR 做二階融合"""
    # 第一層預測
    layer1 = []
    for name, bundle in models_dict.items():
        proba = bundle["model"].predict_proba(X_val)
        layer1.append(proba)
    X_stack = np.hstack(layer1)
    # 用真實 y 建 LR
    true_y = np.column_stack([meta_list[t] for t in models_dict.keys()])
    # 這裡示範單目標，實際可延伸
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_stack, true_y.ravel())
    return lr, X_stack

def train_validate_split(X, y, groups, test_size=0.2, random_state=42):
    from sklearn.model_selection import GroupShuffleSplit

    # Log the length of groups before splitting
    # print(f"Total number of unique groups: {len(groups)}")

    # Debug: Print the first few values of groups
    # print(f"First 10 values in groups: {groups[:10]}")

    # Check unique values in groups
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    # print(f"Number of unique groups: {len(unique_groups)}")
    # logging.info(f"Number of unique groups: {len(unique_groups)}")
    # print(f"Group counts: {dict(zip(unique_groups, group_counts))}")
    # logging.info(f"Group counts: {dict(zip(unique_groups, group_counts))}")


    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    # Ensure unique_id is not repeated
    train_unique_ids = np.unique(groups[train_idx])
    val_unique_ids = np.unique(groups[val_idx])

    if set(train_unique_ids).intersection(set(val_unique_ids)):
        raise ValueError("Data leakage detected: Some unique_ids are in both training and validation sets.")

    # Save unique_ids for debugging
    np.savetxt("train_ids_from_split.txt", train_unique_ids, fmt="%d")
    np.savetxt("val_ids_from_split.txt", val_unique_ids, fmt="%d")

    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'groups_train': groups[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'groups_val': groups[val_idx]
    }

def check_data_leakage(train_ids, val_ids):
    """Check for data leakage between training and validation datasets."""
    train_set = set(train_ids)
    val_set = set(val_ids)

    # Find intersection
    leakage = train_set.intersection(val_set)
    if leakage:
        print("Data leakage detected! Overlapping IDs:", leakage)
        logging.error(f"Data leakage detected! Overlapping IDs: {leakage}")
        return True
    else:
        print("No data leakage detected.")
        logging.info("No data leakage detected.")
        return False

def evaluate_model(model, data_dict, target_info):
    """Evaluate model performance on validation set"""
    model.fit(
        data_dict['X_train'], 
        data_dict['y_train'],
        eval_set=[(data_dict['X_val'], data_dict['y_val'])],
        eval_metric='auc' if target_info["type"] == "bin" else 'multi_logloss',
        callbacks=[lgb.early_stopping(50)]
    )
    
    proba = model.predict_proba(data_dict['X_val'])
    
    if target_info["type"] == "bin":
        score = roc_auc_score(data_dict['y_val'], proba[:, 1])
    else:
        # Convert validation labels to one-hot encoding
        classes = np.unique(data_dict['y_train'])
        y_val_onehot = np.zeros((len(data_dict['y_val']), len(classes)))
        for i, cls in enumerate(classes):
            y_val_onehot[:, i] = (data_dict['y_val'] == cls).astype(int)
            
        score = roc_auc_score(
            y_val_onehot,
            proba,
            multi_class="ovr", 
            average="micro"
        )
    
    return score, model

def evaluate_validation_set(data_dict, models_dict, target_info):
    """Calculate ROC AUC scores for validation data using same logic as evaluate_predictions"""
    scores = {}
    
    for target_name, meta in target_info.items():
        model = models_dict[target_name]['model']
        scaler = models_dict[target_name]['scaler']
        X_val_scaled = scaler.transform(data_dict['X_val'])
        proba = model.predict_proba(X_val_scaled)
        
        if meta["type"] == "bin":
            # 二元分類 - 轉換為 0/1
            true_vals = (data_dict['y_val'][target_name] == 1).astype(int)
            pred_vals = proba[:, 1]  # 使用正類的概率
            score = roc_auc_score(true_vals, pred_vals)
        else:
            # 多分類 - 使用 one-hot 編碼
            if target_name == "play years":
                true_vals = pd.get_dummies(data_dict['y_val'][target_name])
                pred_vals = pd.DataFrame(proba, columns=range(3))
            else:  # level
                true_vals = pd.get_dummies(data_dict['y_val'][target_name])
                pred_vals = pd.DataFrame(proba, columns=[2,3,4,5])
            
            score = roc_auc_score(
                true_vals, pred_vals,
                multi_class="ovr",
                average="micro"
            )
            
        scores[target_name] = score
        print(f"{target_name} ROC AUC: {score:.4f}")
        logging.info(f"{target_name} ROC AUC: {score:.4f}")
    
    avg_score = np.mean(list(scores.values()))
    print(f"\nAverage ROC AUC: {avg_score:.4f}")
    logging.info(f"Average ROC AUC: {avg_score:.4f}")
    return scores, avg_score

def check_unique_id_overlap(train_file, val_file):
    """Check if any unique_id exists in both train and validation files."""
    with open(train_file, 'r') as f:
        train_ids = set(map(int, f.readlines()))

    with open(val_file, 'r') as f:
        val_ids = set(map(int, f.readlines()))

    overlap = train_ids.intersection(val_ids)
    if overlap:
        print("Data leakage detected! Overlapping unique_ids:", overlap)
        logging.error(f"Data leakage detected! Overlapping unique_ids: {overlap}")
    else:
        print("No data leakage detected.")
        logging.info("No data leakage detected.")

# Example usage
check_unique_id_overlap("train_ids_from_split.txt", "val_ids_from_split.txt")