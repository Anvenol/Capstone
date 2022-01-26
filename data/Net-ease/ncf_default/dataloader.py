import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
from tqdm import tqdm, trange
from collections import defaultdict
import time
from sklearn.preprocessing import LabelEncoder
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
    ###########
    # mlog_stats = mlog_stats.get_chunk(1000)
    # user_demographics = user_demographics.get_chunk(1000)
    impression_data = impression_data.get_chunk(1000)
    ##########
    '''处理性别'''
    # user_demographics[user_demographics['gender'] == 'unknown'] = np.nan
    # b=user_demographics['gender'].unique()
    # a=user_demographics['gender'].value_counts().count()
    '''用sklearn来转换标注'''
    lbe_user = LabelEncoder()
    user_demographics[['userId']] = user_demographics[['userId']].apply(lambda x: lbe_user.fit_transform(x))
    lbe_mlog = LabelEncoder()
    mlog_stats[['mlogId']] = mlog_stats[['mlogId']].apply(lambda x: lbe_mlog.fit_transform(x))

    lbe = LabelEncoder()
    user_demographics[['province', 'gender']] = user_demographics[['province', 'gender']].apply(
        lambda x: lbe.fit_transform(x))
    user_demographics[['age', 'gender']] = user_demographics[['gender', 'age']]
    user_demographics.rename(columns={'age': 'gender', 'gender': 'age'}, inplace=True)
    user_demographics.iloc[:, 3:] = (user_demographics.iloc[:, 3:] - user_demographics.iloc[:,
                                                                     3:].mean()) / user_demographics.iloc[:, 3:].std()
    print(user_demographics.head())

    mlog_stats[['mlogId']] = mlog_stats[['mlogId']].apply(lambda x: lbe.fit_transform(x))
    mlog_stats.iloc[:, 1:] = (mlog_stats.iloc[:, 1:] - mlog_stats.iloc[:, 1:].mean()) / mlog_stats.iloc[:, 1:].std()
    print(mlog_stats.head())

    '''返还class数量'''
    user_num = user_demographics['userId'].value_counts().count()
    province_num = user_demographics['province'].value_counts().count()
    gender_num = user_demographics['gender'].value_counts().count()
    user_int_num = 4
    mlog_int_num = 9
    mlog_num = mlog_stats['mlogId'].value_counts().count()

    impression_data[['userId']] = impression_data[['userId']].apply(lambda x: lbe_user.transform(x))
    impression_data[['mlogId']] = impression_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))
    impression_data['isClick'].fillna(0)

    all_user_data = impression_data['userId'].apply(
        lambda x: user_demographics[user_demographics['userId'] == x].values[0])
    all_item_data = impression_data['mlogId'].apply(lambda x: mlog_stats[mlog_stats['mlogId'] == x].values[0])

    all_user_data = np.stack(all_user_data.values)
    all_item_data = np.stack(all_item_data.values)
    isclick = impression_data["isClick"].values.tolist()

    user_train, user_test, item_train, item_test, y_train, y_test = train_test_split(all_user_data, all_item_data,
                                                                                     isclick, test_size=0.33,
                                                                                     random_state=42)

    # print(x_train[:5])
    # print(x_test[:5])
    # print(y_train[:5])
    # print(y_test[:5])
    # print(len(x_train))
    # print(len(x_test))
    # print(len(y_train))
    # print(len(y_test))

    params.user_num = user_num
    params.province_num = province_num
    params.gender_num = gender_num
    params.mlog_num = mlog_num
    params.user_int_num = user_int_num
    params.mlog_int_num = mlog_int_num
    return all_user_data, all_item_data, isclick, user_train, user_test, item_train, item_test, \
           y_train, y_test, params


class TrainSet(data.Dataset):
    def __init__(self, user_features, item_features, values):
        super(TrainSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.user_features = user_features
        self.item_features = item_features
        self.labels = values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_cat = self.user_features[idx, :3].astype(int)
        user_num = self.user_features[idx, 3:].astype(np.float32)
        item_cat = self.item_features[idx, 0].astype(int)
        item_num = self.item_features[idx, 1:].astype(np.float32)
        label = self.labels[idx]
        return user_cat, user_num, item_cat, item_num, label


class TestSet(data.Dataset):
    def __init__(self, user_features, item_features, values, user_features_all,
                 item_features_all, values_all, user_num, mlog_num, test_ng):
        super(TestSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.user_features = user_features[values == 1][:,0]
        self.item_features = item_features[values == 1][:,0]
        self.train_mat = sp.dok_matrix((user_num, mlog_num), dtype=np.float32)
        values_all = np.array(values_all)
        user_true = user_features_all[values_all == 1][:, 0]
        item_true = item_features_all[values_all == 1][:, 0]
        for i in range(user_true.shape[0]):
            self.train_mat[user_true[i], item_true[i]] = 1

        self.mlog_num = mlog_num
        self.test_ng = test_ng
        self.item_id_all = item_features_all[:,0]

    def __len__(self):
        return (self.test_ng + 1) * self.user_features.shape[0]

    def ng_sample(self):
        self.user_fill = []
        self.item_fill = []
        self.labels_fill = []
        for x in range(self.user_features.shape[0]):
            self.user_ng = []
            self.item_ng = []
            for t in range(self.test_ng):
                j = np.random.randint(self.item_id_all.shape[0])
                while (self.user_features[x], self.item_id_all[j]) in self.train_mat:
                    j = np.random.randint(self.item_id_all.shape[0])
                self.user_ng.append(self.user_features[0][x])
                self.item_ng.append(self.item_features[0][j])

            labels_ps = [1]
            labels_ng = [0 for _ in range(self.test_ng)]
            self.user_fill += self.user_features[0][x] + self.user_ng
            self.item_fill += self.item_features[0][x] + self.item_ng
            self.labels_fill += labels_ps + labels_ng

    def __getitem__(self, idx):
        user_cat = self.user_fill[idx, :3].astype(int)
        user_num = self.user_fill[idx, 3:].astype(np.float32)
        item_cat = self.item_fill[idx, 0].astype(int)
        item_num = self.item_fill[idx, 1:].astype(np.float32)
        if user_num is None:
            raise ValueError
        if user_cat is None:
            raise ValueError
        if item_cat is None:
            raise ValueError
        if item_num is None:
            raise ValueError
        return user_cat, user_num, item_cat, item_num

# class TrainSet(data.Dataset):
#     def __init__(self, features, num_item, train_mat=None, num_ng=0):
#         super(TrainSet, self).__init__()
#         """ Note that the labels are only useful when training, we thus
#             add them in the ng_sample() function.
#         """
#         self.features_ps = features
#         self.num_item = num_item
#         self.train_mat = train_mat
#         self.num_ng = num_ng
#         self.labels = [0 for _ in range(len(features))]
#
#     def ng_sample(self):
#         self.features_ng = []
#         for x in self.features_ps:
#             u = x[0]
#             for t in range(self.num_ng):
#                 j = np.random.randint(self.num_item)
#                 while (u, j) in self.train_mat:
#                     j = np.random.randint(self.num_item)
#                 self.features_ng.append([u, j])
#
#         labels_ps = [1 for _ in range(len(self.features_ps))]
#         labels_ng = [0 for _ in range(len(self.features_ng))]
#         self.features_fill = self.features_ps + self.features_ng
#         self.labels_fill = labels_ps + labels_ng
#
#     def __len__(self):
#         return (self.num_ng + 1) * len(self.features_ps)
#
#     def __getitem__(self, idx):
#         user = self.features_fill[idx][0]
#         item = self.features_fill[idx][1]
#         label = self.labels_fill[idx]
#         return user, item, label


# class TestSet(data.Dataset):
#     def __init__(self, features, num_item, num_ng=0):
#         super(TestSet, self).__init__()
#         """ Note that the labels are only useful when training, we thus
#             add them in the ng_sample() function.
#         """
#         self.features_ps = features
#         self.num_item = num_item
#         self.num_ng = num_ng
#
#     def __len__(self):
#         return (self.num_ng + 1) * len(self.features_ps)
#
#     def __getitem__(self, idx):
#         user = self.features_ps[idx][0]
#         item = self.features_ps[idx][1]
#         return user, item
