import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
from tqdm import tqdm, trange
from collections import defaultdict
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

from numba import jit

pd.set_option('display.max_columns', None)


def load_all(params):
    """ We load all the three file here to save time in each epoch. """
    mlog_stats = pd.read_csv(
        os.path.join('data', params.data_dir, params.mlog_stats))
    user_demographics = pd.read_csv(
        os.path.join('data', params.data_dir, params.user_demographics))
    impression_data = pd.read_csv(
        os.path.join('data', params.data_dir, params.impression_data), iterator=True)
    mlog_demographics = pd.read_csv(
        os.path.join('data', params.data_dir, params.mlog_demographics))
    creator_demographics = pd.read_csv(
        os.path.join('data', params.data_dir, params.creator_demographics))
    creator_stats = pd.read_csv(
        os.path.join('data', params.data_dir, params.creator_stats))
    ###########
    # mlog_stats = mlog_stats.get_chunk(1000)
    # user_demographics = user_demographics.get_chunk(1000)
    impression_data = impression_data.get_chunk(10000)
    '''Impression'''
    def get_test_data(x):
        df = x.sort_values(by='impressTime', ascending=True)
        return df.iloc[-1, :]

    def get_train_data(x):
        df = x.sort_values(by='impressTime', ascending=True)
        return df.iloc[:len(x) - 1, :]

    '''user'''
    # user_id = impression_data['userId'].unique()
    # user_demographics = pd.merge(user_demographics, pd.Series(user_id, name='userId'), how='inner').iloc[:, :]
    # group1 = impression_data.groupby('userId').sum()
    # total_num = group1.loc[:, ['isClick', 'isLike', 'isComment', 'isIntoPersonalHomepage',
    #                            'isShare', 'isViewComment', 'mlogViewTime']]
    # total_num.columns = [i + '_total' for i in total_num.columns]
    # total_num['clickto_comment'] = total_num['isComment_total'] / total_num['isClick_total']
    # total_num['clickto_like'] = total_num['isLike_total'] / total_num['isClick_total']
    # total_num['clickto_homepage'] = total_num['isIntoPersonalHomepage_total'] / total_num['isClick_total']
    # total_num['clickto_share'] = total_num['isShare_total'] / total_num['isClick_total']
    # total_num['clickto_viewcomment'] = total_num['isViewComment_total'] / total_num['isClick_total']
    # user_data = pd.merge(user_demographics, total_num, on='userId', how='left')
    user_data = user_demographics
    user_data[['age', 'gender']] = user_data[['gender', 'age']]
    user_data.rename(columns={'age': 'gender', 'gender': 'age'}, inplace=True)
    user_data.loc[:, 'gender'] = user_data.loc[:, 'gender'].fillna('unknown')

    '''mlog'''
    # mlog_id = impression_data['mlogId'].unique()
    # mlog_stats = pd.merge(mlog_stats, pd.Series(mlog_id, name='mlogId'), how='inner').iloc[:, :]
    mlog_data = pd.merge(mlog_stats, mlog_demographics, on='mlogId', how='left')
    mlog_data = pd.merge(mlog_data, creator_demographics, on='creatorId', how='left')
    # cre_num1 = creator_stats.groupby('creatorId').sum()['PushlishMlogCnt'].reset_index().rename(columns={'PushlishMlogCnt': 'PushlishMlogCnt_total'})
    # cre_num2 = creator_stats.groupby('creatorId').mean()['PushlishMlogCnt'].reset_index().rename(columns={'PushlishMlogCnt': 'PushlishMlogCnt_mean'})
    # mlog_data = pd.merge(mlog_data, cre_num1, on='creatorId', how='left')
    # mlog_data = pd.merge(mlog_data, cre_num2, on='creatorId', how='left')
    mlog_data.drop(columns=['creatorId', 'songId', 'artistId', 'contentId'], inplace=True)
    mlog_data[['dt', 'gender']] = mlog_data[['gender', 'dt']]
    mlog_data.rename(columns={'dt': 'gender', 'gender': 'dt'}, inplace=True)
    # mlog_data = mlog_stats

    '''用sklearn来转换标注'''
    lbe_user = LabelEncoder()
    user_data[['userId']] = user_data[['userId']].apply(lambda x: lbe_user.fit_transform(x))
    lbe_mlog = LabelEncoder()
    mlog_data[['mlogId']] = mlog_data[['mlogId']].apply(lambda x: lbe_mlog.fit_transform(x))

    lbe = LabelEncoder()
    user_data[['province', 'gender']] = user_data[['province', 'gender']].apply(lambda x: lbe.fit_transform(x))
    mlog_data[['gender']] = mlog_data[['gender']].apply(lambda x: lbe.fit_transform(x))

    '''返还class数量'''
    mlog_num = mlog_data['mlogId'].value_counts().count()
    user_num = user_data['userId'].value_counts().count()
    user_cat_num = 3
    user_int_num = user_data.shape[1] - user_cat_num
    mlog_cat_num = 2
    mlog_int_num = mlog_data.shape[1] - mlog_cat_num

    '''标准化'''
    # user_data.iloc[:, user_cat_num:].fillna(user_data.iloc[:, user_cat_num:].mean(),inplace=True)
    # mlog_data.iloc[:, mlog_cat_num:].fillna(mlog_data.iloc[:, mlog_cat_num:].mean(),inplace=True)
    user_data.iloc[:, user_cat_num:] = user_data.iloc[:, user_cat_num:].fillna(0)
    mlog_data.iloc[:, mlog_cat_num:] = mlog_data.iloc[:, mlog_cat_num:].fillna(0)

    # user_data.iloc[:, user_cat_num:] = (user_data.iloc[:, user_cat_num:] - user_data.iloc[:,
    #                                                                        user_cat_num:].mean()) / user_data.iloc[:,
    #                                                                                                 user_cat_num:].std()
    # mlog_data.iloc[:, mlog_cat_num:] = (mlog_data.iloc[:, mlog_cat_num:] - mlog_data.iloc[:,
    #                                                                        mlog_cat_num:].mean()) / mlog_data.iloc[:,
    #                                                                                                 mlog_cat_num:].std()
    max_abs_scaler = MaxAbsScaler()
    user_data.iloc[:, user_cat_num:] = max_abs_scaler.fit_transform(user_data.iloc[:, user_cat_num:])
    mlog_data.iloc[:, mlog_cat_num:] = max_abs_scaler.fit_transform(mlog_data.iloc[:, mlog_cat_num:])
    ''''''
    user_cat_dims = []
    for i in range(user_cat_num):
        user_cat_dims.append(user_data.iloc[:, i].value_counts().count())
    mlog_cat_dims = []
    for i in range(mlog_cat_num):
        mlog_cat_dims.append(mlog_data.iloc[:, i].value_counts().count())

    ''''''
    leu_dict = dict(zip(lbe_user.classes_, lbe_user.transform(lbe_user.classes_)))
    impression_data['userId'] = impression_data['userId'].apply(lambda x: leu_dict.get(x, 'Unknown'))
    impression_data = impression_data[impression_data['userId'] != 'Unknown']
    impression_data['userId'] = impression_data['userId'].astype(dtype='int64')

    lem_dict = dict(zip(lbe_mlog.classes_, lbe_mlog.transform(lbe_mlog.classes_)))
    impression_data['mlogId'] = impression_data['mlogId'].apply(lambda x: lem_dict.get(x, 'Unknown'))
    impression_data = impression_data[impression_data['mlogId'] != 'Unknown']
    impression_data['mlogId'] = impression_data['mlogId'].astype(dtype='int64')

    test_data = impression_data.groupby('userId').apply(get_test_data)
    train_data = impression_data.groupby('userId').apply(get_train_data)
    # train_data[['userId']] = train_data[['userId']].apply(lambda x: lbe_user.transform(x))
    # train_data[['mlogId']] = train_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))
    train_data['isClick'].fillna(0)
    train_data.dropna(axis=0, how='any', inplace=False)
    train_user_data = train_data['userId'].apply(lambda x: user_data[user_data['userId'] == x].values[0]).fillna(0)
    train_item_data = train_data['mlogId'].apply(lambda x: mlog_data[mlog_data['mlogId'] == x].values[0]).fillna(0)
    user_train = np.stack(train_user_data.values)
    item_train = np.stack(train_item_data.values)
    y_train = train_data["isClick"].values.tolist()

    # test_data[['userId']] = test_data[['userId']].apply(lambda x: lbe_user.transform(x))
    # test_data[['mlogId']] = test_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))
    test_data['isClick'].fillna(0)
    test_data.dropna(axis=0, how='any', inplace=False)
    test_user_data = test_data['userId'].apply(lambda x: user_data[user_data['userId'] == x].values[0]).fillna(0)
    test_item_data = test_data['mlogId'].apply(lambda x: mlog_data[mlog_data['mlogId'] == x].values[0]).fillna(0)
    user_test = np.stack(test_user_data.values)
    item_test = np.stack(test_item_data.values)
    y_test = test_data["isClick"].values.tolist()

    all_user_data = np.vstack((user_train, user_test))
    all_item_data = np.vstack((item_train, item_test))
    isclick = y_train + y_test

    ''''''
    # impression_data[['userId']] = impression_data[['userId']].apply(lambda x: lbe_user.transform(x))
    # impression_data[['mlogId']] = impression_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))
    # impression_data['isClick'].fillna(0)
    # impression_data.dropna(axis=0, how='any', inplace=False)

    # all_user_data = impression_data['userId'].apply(
    #     lambda x: user_data[user_data['userId'] == x].values[0]).fillna(0)
    # all_item_data = impression_data['mlogId'].apply(lambda x: mlog_data[mlog_data['mlogId'] == x].values[0]).fillna(0)

    # all_user_data = np.stack(all_user_data.values)
    # all_item_data = np.stack(all_item_data.values)
    # isclick = impression_data["isClick"].values.tolist()

    user_train, user_test, item_train, item_test, y_train, y_test = train_test_split(all_user_data, all_item_data,
                                                                                     isclick, test_size=0.33,
                                                                                     random_state=42)

    params.user_num = user_num
    params.mlog_num = mlog_num
    params.user_cat_dims = user_cat_dims
    params.mlog_cat_dims = mlog_cat_dims
    params.user_int_num = user_int_num
    params.mlog_int_num = mlog_int_num
    params.user_cat_num = user_cat_num
    params.mlog_cat_num = mlog_cat_num
    return all_user_data, all_item_data, isclick, user_train, user_test, item_train, item_test, \
           y_train, y_test, params


class TestSet(data.Dataset):
    def __init__(self, user_features, item_features, values, user_cat_num, mlog_cat_num):
        super(TestSet, self).__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.user_cat_num = user_cat_num
        self.mlog_cat_num = mlog_cat_num
        self.labels = values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_cat = self.user_features[idx, :self.user_cat_num].astype(int)
        user_num = self.user_features[idx, self.user_cat_num:].astype(np.float32)
        item_cat = self.item_features[idx, :self.mlog_cat_num].astype(int)
        item_num = self.item_features[idx, self.mlog_cat_num:].astype(np.float32)
        return user_cat, user_num, item_cat, item_num


class TrainSet(data.Dataset):
    def __init__(self, user_features, item_features, values, user_features_all,
                 item_features_all, values_all, user_num, mlog_num, test_ng, user_cat_num, mlog_cat_num):
        super(TrainSet, self).__init__()
        values = np.array(values)
        self.user_features = user_features
        self.item_features = item_features
        self.user_positive_features = user_features[values == 1]
        self.item_positive_features = item_features[values == 1]
        self.train_mat = sp.dok_matrix((user_num, mlog_num), dtype=np.float32)
        values_all = np.array(values_all)
        user_true = user_features_all[values_all == 1][:, 0]
        item_true = item_features_all[values_all == 1][:, 0]
        for i in range(user_true.shape[0]):
            self.train_mat[user_true[i], item_true[i]] = 1

        self.mlog_num = mlog_num
        self.test_ng = test_ng
        self.item_id_all = item_features_all
        self.user_cat_num = user_cat_num
        self.mlog_cat_num = mlog_cat_num

    def __len__(self):
        return (self.test_ng + 1) * self.user_positive_features.shape[0]

    def ng_sample(self):
        self.labels_fill = []
        for x in range(self.user_positive_features.shape[0]):
            self.user_ng = []
            self.item_ng = []
            for t in range(self.test_ng):
                j = np.random.randint(self.item_id_all.shape[0])
                while (self.user_positive_features[x, 0], self.item_id_all[j, 0]) in self.train_mat:
                    j = np.random.randint(self.item_id_all.shape[0])
                self.user_ng.append(self.user_positive_features[x, :])
                self.item_ng.append(self.item_id_all[j, :])

            labels_ps = [1]
            labels_ng = [0 for _ in range(self.test_ng)]
            if x == 0:
                self.user_fill = np.vstack((self.user_positive_features[x, :], np.stack(self.user_ng)))
                self.item_fill = np.vstack((self.item_positive_features[x, :], np.stack(self.item_ng)))
            else:
                self.user_fill = np.vstack((self.user_fill, self.user_positive_features[x, :], np.stack(self.user_ng)))
                self.item_fill = np.vstack((self.item_fill, self.item_positive_features[x, :], np.stack(self.item_ng)))

            self.labels_fill += labels_ps + labels_ng

    def __getitem__(self, idx):
        user_cat = self.user_fill[idx, :self.user_cat_num].astype(int)
        user_num = self.user_fill[idx, self.user_cat_num:].astype(np.float32)
        item_cat = self.item_fill[idx, :self.mlog_cat_num].astype(int)
        item_num = self.item_fill[idx, self.mlog_cat_num:].astype(np.float32)
        label = self.labels_fill[idx]
        return user_cat, user_num, item_cat, item_num, label