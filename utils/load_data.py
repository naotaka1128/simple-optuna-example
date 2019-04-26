import pandas as pd
import glob

def load_data(scaled=False, test_mode=0):
    df_train, df_test = load_data_base(scaled, test_mode)

    if test_mode == 1:
        df_train = df_train.loc[:20000, :]
        df_test = df_test.loc[:20000, :]

    return df_train, df_test


def load_data_base(scaled=False, test_mode=False):
    if scaled:
        df_train = pd.read_pickle(
            './data/20190307_base/df_train_scaled.pickle')
        df_test = pd.read_pickle(
            './data/20190307_base/df_test_scaled.pickle')
    else:
        df_train = pd.read_pickle(
            './data/20190307_base/df_train.pickle')
        df_test = pd.read_pickle(
            './data/20190307_base/df_test.pickle')

    return df_train, df_test
