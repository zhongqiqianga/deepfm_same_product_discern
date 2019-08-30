import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from v20190708.rank.example import config
from v20190708.rank.DeepFM import DeepFM
# from v20190708.rank.raw_deepfm import DeepFM




def _run_base_model_dfm( dfm_params):
    dfm_params["feature_size"] = 239083
    dfm_params["field_size"] = 54
    Xi_train=pd.read_csv("/Users/looker/project/xmodel/v20190708/rank/example/output/Xi_train_split0.csv",header=None)
    Xv_train=pd.read_csv("/Users/looker/project/xmodel/v20190708/rank/example/output/Xv_train_split0.csv",header=None)
    y_train=pd.read_csv("/Users/looker/project/xmodel/v20190708/rank/example/output/y_train_split0.csv",header=None,names=['if_click'])
    y_train=pd.Series(y_train['if_click'].values)
    print(Xi_train.shape)
    print(Xi_train.head(10))
    print(y_train.shape)
    print(y_train.head(10))


    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,random_state=config.RANDOM_SEED).split(Xv_train.values, y_train.values))
    y_train_meta = np.zeros((Xi_train.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    auc_results_cv = np.zeros(len(folds), dtype=float)
    results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    print(y_train.values.shape)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train.values.tolist(), train_idx), _get(Xv_train.values.tolist(), train_idx), _get(y_train.values.tolist(), train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train.values.tolist(), valid_idx), _get(Xv_train.values.tolist(), valid_idx), _get(y_train.values.tolist(), valid_idx)
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        auc_results_cv[i] = roc_auc_score(y_valid_, y_train_meta[valid_idx])#求出auc
        results_epoch_train[i] = dfm.train_result
        results_epoch_valid[i] = dfm.valid_result

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, auc_results_cv.mean(), auc_results_cv.std()))
    return y_train_meta





# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 2048,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    # "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm = _run_base_model_dfm(dfm_params)

# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm = _run_base_model_dfm( fm_params)

# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn = _run_base_model_dfm( dnn_params)
