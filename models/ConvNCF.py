import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from typing import List, Tuple

import utils

logger = logging.getLogger('RMD.ConvNCF')


def train(params, evaluate_metrics, train_loader, test_loader):
    combined_model = Net(params, model='ConvNCF')
    if torch.cuda.is_available():
        combined_model.cuda()
    logger.info('Training model...')
    train_single_model(combined_model, params, evaluate_metrics, train_loader, test_loader, 'ConvNCF')
    return combined_model


def train_single_model(model, params, evaluate_metrics, train_loader, test_loader, model_name):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.log_output:
        writer = SummaryWriter(log_dir=os.path.join(params.plot_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    count, best_hr, best_epoch, best_ndcg = 0, 0, -1, 0

    loss_summary = np.zeros(params.num_batches * params.epochs)
    HR_summary = np.zeros(params.epochs)
    NDCG_summary = np.zeros(params.epochs)

    for epoch in trange(params.epochs):
        model.train()
        if epoch % 10 == 0:
            test_loader.dataset.ng_sample()

        for batch in train_loader:
            user_cat, user_num, item_cat, item_num, label = map(lambda x: x.to(params.device), batch)

            model.zero_grad()
            prediction = model(user_cat, user_num, item_cat, item_num)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()
            loss_summary[count] = loss.item()
            if params.log_output:
                writer.add_scalar(f'{model_name}/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate_metrics(model, test_loader, params.top_k, params.device)
        HR_summary[epoch] = HR
        NDCG_summary[epoch] = NDCG
        if params.log_output:
            writer.add_scalars(f'{model_name}/accuracy', {'HR': np.mean(HR),
                                                          'NDCG': np.mean(NDCG)}, epoch)

        logger.info(f'Epoch {epoch} - HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}')

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            torch.save(model, os.path.join(params.model_dir, f'{model_name}_best.pth'))
            logger.info(f'Epoch {epoch} - found best!')
            utils.save_dict_to_json({"HR": HR, "NDCG": NDCG}, os.path.join(params.model_dir, 'metrics_test_best_weights.json'))

        if epoch % 100 == 99:
            utils.plot_all_loss(loss_summary[:count], 'loss', plot_title='loss_summary',
                                location=os.path.join(params.model_dir, 'figures'))
            utils.plot_all_epoch(HR_summary[:epoch+1], NDCG_summary[:epoch+1], 'metrics', plot_title='metrics_summary',
                                location=os.path.join(params.model_dir, 'figures'))
        # torch.save(model, os.path.join(params.model_dir, f'{model_name}_epoch_{epoch}.pth'))

    if params.log_output:
        writer.close()
    logger.info(f"End training. Best epoch {best_epoch:03d}: HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}")


class Net(nn.Module):
    def __init__(self, params, model):
        super(Net, self).__init__()
        self.dropout = params.dropout
        self.model = model
        factor_num = params.factor_num
        num_layers = params.num_layers
        user_num = params.user_num
        mlog_num = params.mlog_num
        user_int_num = params.user_int_num
        mlog_int_num = params.mlog_int_num
        user_cat_num = params.user_cat_num
        mlog_cat_num = params.mlog_cat_num
        user_cat_dims = params.user_cat_dims
        mlog_cat_dims = params.mlog_cat_dims
        self.embedding_size = 16

        # self.user_embedding1 = nn.Embedding(user_cat_dims[0], 42)
        self.user_embedding2 = nn.Embedding(user_cat_dims[1], 7)
        self.user_embedding3 = nn.Embedding(user_cat_dims[2], 5)
        self.user_embedding4 = nn.Linear(user_int_num, 10)
        # self.mlog_embedding1 = nn.Embedding(mlog_cat_dims[0], 36)
        self.mlog_embedding2 = nn.Linear(mlog_int_num, 20)
        self.mlog_embedding3 = nn.Embedding(mlog_cat_dims[1], 5)

        self.embed_user = nn.Linear(7 + 5 + 10, self.embedding_size)
        self.embed_item = nn.Linear(20 + 5, self.embedding_size)

        # cnn setting
        self.channel_size = 16
        self.kernel_size = 2
        self.strides = 2
        self.cnn = nn.Sequential(
            # batch_size * 1 * 16 * 16
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )

        # fully-connected layer, used to predict
        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 1)

        # dropout
        self.dropout1 = nn.Dropout(params.dropout)
        self.dropout2 = nn.Dropout(params.dropout)
        self.dropout3 = nn.Dropout(params.dropout)
        self.dropout4 = nn.Dropout(params.dropout)

        self.dropout_feature = params.dropout_feature

    def forward(self, user_cat, user_num, item_cat, item_num):
        """
        for i in range(numerical_feature_start):
          embed_userid = self.user_embedding1(user_cat[:, i])
        """
        # embed_userid = self.user_embedding1(user_cat[:, 0])
        embed_province = self.user_embedding2(user_cat[:, 1])
        embed_gender = self.user_embedding3(user_cat[:, 2])
        embed_user_linear = self.user_embedding4(user_num)
        user = torch.cat((embed_province, embed_gender, embed_user_linear), dim=1)

        # embed_mlogid = self.mlog_embedding1(item_cat[:,0])
        embed_mloggender = self.mlog_embedding3(item_cat[:, 1])
        embed_mlog_linear = self.mlog_embedding2(item_num)
        item = torch.cat((embed_mloggender, embed_mlog_linear), dim=1)

        if self.dropout_feature == 0:
            user_embeddings = self.embed_user(self.dropout1(user))
            item_embeddings = self.embed_item(self.dropout2(item))
            print('no dropout')
        else:
            user_embeddings = self.dropout3(self.embed_user(self.dropout1(user)))
            item_embeddings = self.dropout4(self.embed_item(self.dropout2(item)))
            print('dropout')

        # outer product
        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
        interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

        # cnn
        feature_map = self.cnn(interaction_map)  # output: batch_size * 16 * 1 * 1
        feature_vec = feature_map.view((-1, self.channel_size))

        # fc
        prediction = self.fc1(torch.cat((feature_vec, user_embeddings, item_embeddings), dim=1))
        prediction = self.fc2(prediction).view((-1))

        return prediction
