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
    impression_data = impression_data.get_chunk(1000)
    ##########
    '''处理性别'''
    # user_demographics[user_demographics['gender'] == 'unknown'] = np.nan
    # b=user_demographics['gender'].unique()
    # a=user_demographics['gender'].value_counts().count()
    '''user'''
    group1 = impression_data.groupby('userId').sum()
    total_num = group1.loc[:, ['isClick', 'isLike', 'isComment', 'isIntoPersonalHomepage',
                               'isShare', 'isViewComment', 'mlogViewTime']]
    total_num.columns = [i + '_total' for i in total_num.columns]
    total_num['clickto_comment'] = total_num['isComment_total'] / total_num['isClick_total']
    total_num['clickto_like'] = total_num['isLike_total'] / total_num['isClick_total']
    total_num['clickto_homepage'] = total_num['isIntoPersonalHomepage_total'] / total_num['isClick_total']
    total_num['clickto_share'] = total_num['isShare_total'] / total_num['isClick_total']
    total_num['clickto_viewcomment'] = total_num['isViewComment_total'] / total_num['isClick_total']
    user_data = pd.merge(user_demographics, total_num, on='userId', how='left')
    user_data[['age', 'gender']] = user_data[['gender', 'age']]
    user_data.rename(columns={'age': 'gender', 'gender': 'age'}, inplace=True)
    '''mlog'''
    mlog_data = pd.merge(mlog_stats, mlog_demographics, on='mlogId',how='left')
    mlog_data = pd.merge(mlog_data, creator_demographics, on='creatorId',how='left')
    cre_num1 = creator_stats.groupby('creatorId').sum()['PushlishMlogCnt'].reset_index().rename(columns={'PushlishMlogCnt': 'PushlishMlogCnt_total'})
    cre_num2 = creator_stats.groupby('creatorId').mean()['PushlishMlogCnt'].reset_index().rename(columns={'PushlishMlogCnt': 'PushlishMlogCnt_mean'})
    mlog_data = pd.merge(mlog_data, cre_num1, on='creatorId', how='left')
    mlog_data = pd.merge(mlog_data, cre_num2, on='creatorId', how='left')
    mlog_data.drop(columns=['creatorId','songId','artistId'], inplace=True)
    mlog_data[['dt_x', 'gender']] = mlog_data[['gender', 'dt_x']]
    mlog_data.rename(columns={'dt_x': 'gender', 'gender': 'dt_x'}, inplace=True)
    '''用sklearn来转换标注'''
    lbe_user = LabelEncoder()
    user_data[['userId']] = user_data[['userId']].apply(lambda x: lbe_user.fit_transform(x))
    lbe_mlog = LabelEncoder()
    mlog_data[['mlogId']] = mlog_data[['mlogId']].apply(lambda x: lbe_mlog.fit_transform(x))

    lbe = LabelEncoder()
    user_data[['province', 'gender']] = user_data[['province', 'gender']].apply(lambda x: lbe.fit_transform(x))
    mlog_data[['gender']] = mlog_data[['gender']].apply(lambda x: lbe.fit_transform(x))

    # user_demographics.fillna(user_demographics.mean(), inplace=True)
    # mlog_stats.iloc[:, 1:].fillna(mlog_stats.iloc[:, 1:].mean())

    '''标准化'''
    # sc = StandardScaler()
    # user_demographics.iloc[:, [6]] = sc.fit_transform(user_demographics.iloc[:, [6]])
    # mlog_stats.iloc[:, [1]] = sc.fit_transform(mlog_stats.iloc[:, [1]])
    #
    # mm = MinMaxScaler()
    # rs = RobustScaler()
    # user_demographics.iloc[:, [3,4,5]] = rs.fit_transform(user_demographics.iloc[:, [3,4,5]])
    # mlog_stats.iloc[:, 2:] = rs.fit_transform(mlog_stats.iloc[:, 2:])
    #
    # print(user_demographics.max())
    # print(mlog_stats.max())

    '''返还class数量'''
    # province_num = user_demographics['province'].value_counts().count()
    # gender_num = user_demographics['gender'].value_counts().count()
    mlog_num = mlog_stats['mlogId'].value_counts().count()
    user_num = user_demographics['userId'].value_counts().count()
    user_cat_num = 3
    user_int_num = user_num-user_cat_num
    mlog_cat_num = 2
    mlog_int_num = mlog_num-mlog_cat_num
    '''标准化'''
    user_demographics.iloc[:, user_cat_num:] = (user_demographics.iloc[:, user_cat_num:] - user_demographics.iloc[:,
                                                                     user_cat_num:].mean()) / user_demographics.iloc[:, user_cat_num:].std()
    mlog_stats.iloc[:, mlog_cat_num:] = (mlog_stats.iloc[:, mlog_cat_num:] - mlog_stats.iloc[:, mlog_cat_num:].mean()) / mlog_stats.iloc[:, mlog_cat_num:].std()
    ''''''
    user_cat_dims = []
    for i in range(user_cat_num):
        user_cat_dims.append(user_demographics.iloc[:,i].value_counts().count())
    mlog_cat_dims = []
    for i in range(mlog_cat_num):
        mlog_cat_dims.append(mlog_stats.iloc[:,i].value_counts().count())

    impression_data[['userId']] = impression_data[['userId']].apply(lambda x: lbe_user.transform(x))
    impression_data[['mlogId']] = impression_data[['mlogId']].apply(lambda x: lbe_mlog.transform(x))
    impression_data['isClick'].fillna(0)
    impression_data.dropna(axis=0, how='any', inplace=False)

    all_user_data = impression_data['userId'].apply(
        lambda x: user_demographics[user_demographics['userId'] == x].values[0])
    all_item_data = impression_data['mlogId'].apply(lambda x: mlog_stats[mlog_stats['mlogId'] == x].values[0])

    all_user_data = np.stack(all_user_data.values)
    all_item_data = np.stack(all_item_data.values)
    isclick = impression_data["isClick"].values.tolist()

    user_train, user_test, item_train, item_test, y_train, y_test = train_test_split(all_user_data, all_item_data,
                                                                                     isclick, test_size=0.33,
                                                                                     random_state=42)

    '''标准化'''
    # sc = StandardScaler()
    # mm = MinMaxScaler()
    # rs = RobustScaler()
    # ma = MaxAbsScaler()
    # item_train[:, 1:] = ma.fit_transform(item_train[:, 1:])
    # item_test[:, 1:] = ma.transform(item_test[:, 1:])
    #
    # user_train[:, 3:] = sc.fit_transform(user_train[:, 3:])
    # user_test[:, 3:] = sc.transform(user_test[:, 3:])

    # user_demographics.iloc[:, [6]] = sc.fit_transform(user_demographics.iloc[:, [6]])
    # mlog_stats.iloc[:, [1]] = sc.fit_transform(mlog_stats.iloc[:, [1]])

    # user_demographics.iloc[:, [3,4,5]] = rs.fit_transform(user_demographics.iloc[:, [3,4,5]])
    # mlog_stats.iloc[:, 2:] = rs.fit_transform(mlog_stats.iloc[:, 2:])
    #
    # print(user_demographics.max())
    # print(mlog_stats.max())

    # params.province_num = province_num
    # params.gender_num = gender_num
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


class TrainSet(data.Dataset):
    def __init__(self, user_features, item_features, values, user_cat_num, mlog_cat_num):
        super(TrainSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
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
        label = self.labels[idx]
        return user_cat, user_num, item_cat, item_num, label


class TestSet(data.Dataset):
    def __init__(self, user_features, item_features, values, user_features_all,
                 item_features_all, values_all, user_num, mlog_num, test_ng, user_cat_num, mlog_cat_num):
        super(TestSet, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
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
                while (self.user_positive_features[x,0], self.item_id_all[j,0]) in self.train_mat:
                    j = np.random.randint(self.item_id_all.shape[0])
                self.user_ng.append(self.user_positive_features[x,:])
                self.item_ng.append(self.item_id_all[j,:])

            labels_ps = [1]
            labels_ng = [0 for _ in range(self.test_ng)]
            if x == 0:
                self.user_fill = np.vstack((self.user_positive_features[x,:], np.stack(self.user_ng)))
                self.item_fill = np.vstack((self.item_positive_features[x,:], np.stack(self.item_ng)))
            else:
                self.user_fill = np.vstack((self.user_fill, self.user_positive_features[x,:], np.stack(self.user_ng)))
                self.item_fill = np.vstack((self.item_fill, self.item_positive_features[x,:], np.stack(self.item_ng)))

            self.labels_fill += labels_ps + labels_ng

    def __getitem__(self, idx):
        user_cat = self.user_fill[idx, :self.user_cat_num].astype(int)
        user_num = self.user_fill[idx, self.user_cat_num:].astype(np.float32)
        item_cat = self.item_fill[idx, :self.mlog_cat_num].astype(int)
        item_num = self.item_fill[idx, self.mlog_cat_num:].astype(np.float32)
        return user_cat, user_num, item_cat, item_num