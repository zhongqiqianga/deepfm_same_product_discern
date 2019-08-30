
import pandas as pd
from v20190708.rank.example import config
from v20190708.rank.example.DataReader_2 import FeatureDictionary, DataParser
import os

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


def _run_base_model_dfm(dfTrain, dfTest):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest, numeric_cols=config.NUMERIC_COLS,ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    # Xi_test, Xv_test, uids_test,pids_test = data_parser.parse(df=dfTest)
    print("feature_size"+str(fd.feat_dim))
    print("field_size"+str(len(Xi_train.columns)))
    start=0
    file_size=50000
    end=start+file_size
    file_index=0
    while(end<=Xi_train.shape[0]):
        Xi_train_split=Xi_train[start:end]
        Xv_train_split=Xv_train[start:end]
        y_train_split=y_train[start:end]
        filename_Xi = "Xi_train_split%s.csv" % (str(file_index))
        filename_Xv = "Xv_train_split%s.csv" % (str(file_index))
        filename_y = "y_train_split%s.csv" % (str(file_index))
        _make_submission(Xi_train_split,filename_Xi)
        _make_submission(Xv_train_split,filename_Xv)
        _make_submission(y_train_split,filename_y)
        if(end==Xi_train.shape[0]):
            break
        start=end
        end=end+file_size
        if (end > Xi_train.shape[0]):
            end = Xi_train.shape[0]
        print(Xi_train_split.shape)
        file_index+=1




def _make_submission(content,filename):
    content.to_csv(os.path.join(config.SUB_DIR, filename), index=False, header=None)


# load data
dfTrain, dfTest, X_train, y_train, X_test, uids_test, pids_test, cat_features_indices = _load_data()
_run_base_model_dfm(dfTrain, dfTest)

