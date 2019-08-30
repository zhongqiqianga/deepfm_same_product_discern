import codecs
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from v20190708.rank.example import config
from v20190708.rank.example.DataReader import FeatureDictionary, DataParser
from v20190708.rank.DeepFM import DeepFM
import json

# from v20190708.rank.raw_deepfm import DeepFM


def _load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    cols = [c for c in dfTrain.columns if c not in ["if_click"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]
    X_train = dfTrain[cols].values
    y_train = dfTrain["if_click"].values
    X_test = dfTest[cols].values
    pids_test = dfTest["pid"].values
    uids_test = dfTest["uid"].values
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]
    return dfTrain, dfTest, X_train, y_train, X_test, uids_test, pids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    fw = open("/Users/looker/project/xmodel/v20190708/rank/example/data/embeding_index.txt", 'w+')
    fw.write(str(fd.feat_dict))
    fw.close()
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, uids_test, pids_test = data_parser.parse(df=dfTest)
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    auc_results_cv = np.zeros(len(folds), dtype=float)
    results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)
        auc_results_cv[i] = roc_auc_score(y_valid_, y_train_meta[valid_idx])  # 求出auc
        results_epoch_train[i] = dfm.train_result
        results_epoch_valid[i] = dfm.valid_result
        # dfm.saver.save(dfm.sess, '/Users/looker/project/xmodel/v20190708/rank/example/output/my_test_model')
        print(y_test_meta[:10])
        # 第二种constant_graph=tf.graph_util.convert_variables_to_constants(dfm.sess,dfm.sess.graph_def,['feat_index','feat_value','label','dropout_keep_fm','dropout_keep_deep','train_phase','output'])
        # with tf.gfile.FastGFile('/Users/looker/project/xmodel/v20190708/rank/example/output/test_model.pb', mode='wb') as f:  # 模型的名字是model.pb
        #     f.write(constant_graph.SerializeToString())
        # dfm.sess.run(dfm.in)
        # tf.saved_model.simple_save(dfm.sess, "./model_path",
        # inputs={'feat_index': dfm.feat_index, 'feat-value': dfm.feat_value,"label": dfm.label, 'dropout_keep_fm':
        # dfm.dropout_keep_fm, "dropout_keep_deep": dfm.dropout_keep_deep,"train_phase": dfm.train_phase},outputs={'output':dfm.out})
        # print("模型写入完成")
    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, auc_results_cv.mean(), auc_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, auc_results_cv.mean(), auc_results_cv.std())
    _make_submission(uids_test, pids_test, y_test_meta, filename)
    _plot_fig(results_epoch_train, results_epoch_valid, clf_str)
    return y_train_meta, y_test_meta


def _make_submission(uids, pids, y_pred, filename="submission.csv"):
    pd.DataFrame({"uid": uids, 'pid': pids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png" % model_name)
    plt.close()


# load data
dfTrain, dfTest, X_train, y_train, X_test, uids_test, pids_test, cat_features_indices = _load_data()
# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

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
    "epoch": 1,
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
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# ------------------ FM Model ------------------
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)

# ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
