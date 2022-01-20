import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
from tqdm import tqdm
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
    '''用sklearn来转换标注'''

    lbe_user = LabelEncoder()
    user_demographics[['userId']] = user_demographics[['userId']].apply(lambda x: lbe_user.fit_transform(x))
    lbe_mlog = LabelEncoder()
    mlog_stats[['mlogId']] = mlog_stats[['mlogId']].apply(lambda x: lbe_mlog.fit_transform(x))

    impression_data[['userId']] = impression_data[['userId']].apply(lambda x: lbe_user.transform(x))
    impression_data[['mlogId']] = impression_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))

    impression_data['userId'] = impression_data['userId'].apply(lambda x: user_demographics[user_demographics['userId'] == x].values.tolist()[0])
    impression_data['mlogId'] = impression_data['mlogId'].apply(lambda x: mlog_stats[mlog_stats['mlogId'] == x].values.tolist()[0])
    print(impression_data.head())

    '''循环'''
    # mlog_dic = {}
    # user_dic = defaultdict(int)
    # mlog_length = len(mlog_stats)
    # user_length = len(user_demographics['userId'])
    #
    # start = time.time()
    # end = time.time()
    # print("Running time: %s seconds" % (end - start))
    #
    # for ind, row in tqdm(user_demographics.iterrows()):
    #     user_dic[row['userId']] = ind
    #
    # for ind, row in tqdm(mlog_stats.iterrows()):
    #     mlog_dic[row['mlogId']] = ind

    # mlog_stats_numpy_array = mlog_stats.values
    # user_demographics_numpy_array = user_demographics.values

    # @jit(nopython=True)
    # def generate_for_loop(mlog_stats,user_demographics,mlog_length,user_length):
    #     mlog_dic = {}
    #     user_dic = {}
    #
    #     for i in range(mlog_length):
    #         mlog_dic[mlog_stats[i][0]] = i
    #     for i in range(user_length):
    #         user_dic[user_demographics[i][0]] = i
    #
    #     return mlog_dic, user_dic
    #
    # mlog_dic,user_dic = generate_for_loop(mlog_stats_numpy_array,user_demographics_numpy_array,mlog_length,user_length)

    # for i in range(mlog_length):
    #     mlog_dic[mlog_stats.loc[i][0]] = i
    # for i in range(user_length):
    #     user_dic[user_demographics.loc[i][0]] = i
    ''''''

    # user_array = user_demographics.values
    #
    # mlog_array = mlog_stats.values

    train_data = impression_data[["userId", "mlogId"]].values.tolist()
    isclick = impression_data["isClick"].values.tolist()

    x_train, x_test, y_train, y_test = train_test_split(train_data, isclick, test_size=0.33, random_state=42)
    print(x_train[:5])
    print(x_test[:5])
    print(y_train[:5])
    print(y_test[:5])
    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))

    # train_data[['userId']] = train_data[['userId']]

    # train_mat = sp.dok_matrix((user_length, mlog_length), dtype=np.float32)

    # for ind, row in tqdm(train_data.iterrows()):
    #     train_mat[user_dic[row['userId']], mlog_dic[row['mlogId']]] = row['isClick']

    # for x in tqdm(train_data.values):
    #     train_mat[user_dic[x[2]], mlog_dic[x[1]]] = x[0]

    # user_num = user_length
    # item_num = mlog_length

    return x_train, x_test, y_train, y_test

class TrainSet(data.Dataset):
    def __init__(self, features, values):
        super(TrainSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features = features
        self.labels = values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        item = self.features[idx][1]
        label = self.labels[idx]
        return user, item, label

class TestSet(data.Dataset):
    def __init__(self, features, values):
        super(TestSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features = features
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        item = self.features[idx][1]
        return user, item

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
