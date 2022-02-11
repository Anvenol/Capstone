import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger('RMD.ncf')


def train(params, evaluate_metrics, train_loader, test_loader):
    GMF_model = Net(params, model='GMF')
    MLP_model = Net(params, model='MLP')
    if torch.cuda.is_available():
        GMF_model.cuda()
        MLP_model.cuda()
    logger.info('Pretraining GMF only...')
    train_single_model(GMF_model, params, evaluate_metrics, train_loader, test_loader, 'GMF')
    logger.info('Pretraining MLP only...')
    train_single_model(MLP_model, params, evaluate_metrics, train_loader, test_loader, 'MLP')

    combined_model = Net(params, model='NeuMF-pre', GMF_model=GMF_model, MLP_model=MLP_model)
    if torch.cuda.is_available():
        combined_model.cuda()
    logger.info('Training combined model...')
    train_single_model(combined_model, params, evaluate_metrics, train_loader, test_loader, 'NeuMF-pre')
    return combined_model


def train_single_model(model, params, evaluate_metrics, train_loader, test_loader, model_name):
    loss_fn = nn.BCEWithLogitsLoss()

    if model_name == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=params.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.log_output:
        writer = SummaryWriter(log_dir=os.path.join(params.plot_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    count, best_hr, best_epoch, best_ndcg = 0, 0, -1, 0

    for epoch in trange(params.epochs):
        model.train()
        test_loader.dataset.ng_sample()

        for user_cat, user_num, item_cat, item_num, label in train_loader:
            user_cat = user_cat.to(params.device)
            user_num = user_num.to(params.device)
            item_cat = item_cat.to(params.device)
            item_num = item_num.to(params.device)
            label = label.float().to(params.device)

            model.zero_grad()
            prediction = model(user_cat, user_num, item_cat, item_num)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()
            if params.log_output:
                writer.add_scalar(f'{model_name}/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate_metrics(model, test_loader, params.top_k, params.device)
        if params.log_output:
            writer.add_scalars(f'{model_name}/accuracy', {'HR': np.mean(HR),
                                                          'NDCG': np.mean(NDCG)}, epoch)

        logger.info(f'Epoch {epoch} - HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}')

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            torch.save(model, os.path.join(params.model_dir, f'{model_name}_best.pth'))
        torch.save(model, os.path.join(params.model_dir, f'{model_name}_epoch_{epoch}.pth'))

    if params.log_output:
        writer.close()
    logger.info(f"End training. Best epoch {best_epoch:03d}: HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}")


class Net(nn.Module):
    def __init__(self, params, model, GMF_model=None, MLP_model=None):
        """
		Args:
		    paams: dictionary of all parameters
            model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre'
            GMF_model: pre-trained GMF weights
            MLP_model: pre-trained MLP weights
		"""
        super(Net, self).__init__()
        self.dropout = params.dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        # self.custom_embedding1 = nn.Embedding(num_class, embed_size)
        factor_num = params.factor_num
        num_layers = params.num_layers
        user_num = params.user_num
        mlog_num = params.mlog_num
        province_num = params.province_num
        gender_num = params.gender_num
        user_int_num = params.user_int_num
        mlog_int_num = params.mlog_int_num

        self.user_embedding1 = nn.Embedding(user_num, 10)
        self.user_embedding2 = nn.Embedding(province_num, 10)
        self.user_embedding3 = nn.Embedding(gender_num, 10)
        self.user_embedding4 = nn.Linear(user_int_num,10)
        self.mlog_embedding1 = nn.Embedding(mlog_num, 10)
        self.mlog_embedding2 = nn.Linear(mlog_int_num,10)

        '''
        self.user_cats = nn.ModuleList([nn.Embedding(user_num, 10),
        nn.Embedding(province_num, 10),
        nn.Embedding(gender_num, 10)
        ])
        # in forward: self.user_cats[0] to run the first embedding layer
        '''

        self.embed_user_GMF = nn.Linear(40, factor_num)
        self.embed_item_GMF = nn.Linear(20, factor_num)
        self.embed_user_MLP = nn.Linear(
            40, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Linear(
            20, factor_num * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user_cat, user_num, item_cat, item_num):
        # self.custom_embedding1(user[:, 5])
        """
        for i in range(numerical_feature_start):
          embed_userid = self.user_embedding1(user_cat[:, i])
        """
        embed_userid = self.user_embedding1(user_cat[:, 0])
        embed_province = self.user_embedding2(user_cat[:, 1])
        embed_gender = self.user_embedding3(user_cat[:, 2])
        embed_user_linear = self.user_embedding4(user_num)
        user = torch.cat((embed_userid, embed_province, embed_gender, embed_user_linear), dim=1)

        embed_mlogid = self.mlog_embedding1(item_cat)
        embed_mlog_linear = self.mlog_embedding2(item_num)
        item = torch.cat((embed_mlogid, embed_mlog_linear), dim=1)

        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
