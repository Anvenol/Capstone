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

logger = logging.getLogger('RMD.VSN_ConvNCF')


def train(params, evaluate_metrics, train_loader, val_loader, test_loader):
    combined_model = Net(params, model='VSN_ConvNCF')
    if torch.cuda.is_available():
        combined_model.cuda()
    logger.info('Training model...')
    train_single_model(combined_model, params, evaluate_metrics, train_loader, val_loader, test_loader, 'VSN_ConvNCF')
    return combined_model


def train_single_model(model, params, evaluate_metrics, train_loader, val_loader, test_loader, model_name):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    if params.log_output:
        writer = SummaryWriter(log_dir=os.path.join(params.plot_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    count, best_val_loss, best_hr, best_epoch, best_ndcg = 0, 1e8, 0, -1, 0

    loss_summary = np.zeros(params.num_batches * params.epochs)
    val_loss_summary = np.zeros(params.epochs)
    HR_summary = np.zeros(params.epochs)
    NDCG_summary = np.zeros(params.epochs)

    user_weights_all = np.zeros(params.epochs, params.user_int_num + params.user_cat_num - 1)
    item_weights_all = np.zeros(params.epochs, params.mlog_int_num + params.mlog_cat_num - 1)
    item_header_list = ['hi'] * (params.user_int_num + params.user_cat_num - 1)
    user_header_list = ['hi'] * (params.epochs, params.mlog_int_num + params.mlog_cat_num - 1)

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
        with torch.no_grad():
            loss_val_epoch = np.zeros(params.num_val_batches)
            for val_count, batch in enumerate(val_loader):
                user_cat, user_num, item_cat, item_num, label = map(lambda x: x.to(params.device), batch)
                prediction, user_weights, item_weights = model(user_cat, user_num, item_cat,
                                                               item_num, return_weights=True)
                print('user_weights: ', user_weights.shape)
                print('item_weights: ', item_weights.shape)
                print('user_weights_all: ', user_weights_all.shape)
                print('item_weights_all: ', item_weights_all.shape)
                loss_val_epoch[val_count] = loss_fn(prediction, label).item()
                user_weights_all[epoch] = user_weights.data.cpu().numpy().mean(0)
                item_weights_all[epoch] = item_weights.data.cpu().numpy().mean(0)

            val_loss = np.mean(loss_val_epoch)
            val_loss_summary[epoch] = val_loss

        HR, NDCG = evaluate_metrics(model, test_loader, params.top_k, params.device)
        HR_summary[epoch] = HR
        NDCG_summary[epoch] = NDCG
        if params.log_output:
            writer.add_scalars(f'{model_name}/accuracy', {'HR': np.mean(HR),
                                                          'NDCG': np.mean(NDCG)}, epoch)

        logger.info(f'Epoch {epoch} - val_loss: {val_loss:.3f}, HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}')

        if val_loss < best_val_loss:
            best_val_loss, best_hr, best_ndcg, best_epoch = val_loss, HR, NDCG, epoch
            torch.save(model, os.path.join(params.model_dir, f'{model_name}_best.pth'))
            logger.info(f'Epoch {epoch} - found best! Best HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}')
            utils.save_dict_to_json({'val_loss': val_loss, 'HR': HR, 'NDCG': NDCG},
                                    os.path.join(params.model_dir, 'metrics_test_best_weights.json'))

        if epoch % 100 == 99:
            utils.plot_all_loss(loss_summary[:count], 'loss', plot_title=params.plot_title,
                                location=os.path.join(params.model_dir, 'figures'))
            utils.plot_all_epoch(val_loss_summary[:epoch+1], HR_summary[:epoch+1], NDCG_summary[:epoch+1],
                                 'metrics', plot_title=params.plot_title,
                                 location=os.path.join(params.model_dir, 'figures'))
            utils.plot_weights(item_weights_all, user_weights_all, item_header_list, user_header_list,
                               HR_summary, epoch)
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

        self.embed_user = VSN(d_hidden=self.embedding_size, n_vars=user_int_num, cat_vars=user_cat_num - 1,
                              cat_dims=user_cat_dims[1:])
        self.embed_item = VSN(d_hidden=self.embedding_size, n_vars=mlog_int_num, cat_vars=mlog_cat_num - 1,
                              cat_dims=mlog_cat_dims[1:])

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

    def forward(self, user_cat, user_num, item_cat, item_num, return_weights=False):
        user_embeddings, user_weights = self.embed_user.forward(variables=user_num,
                                                                cat_variables=user_cat[:, 1:])
        item_embeddings, item_weights = self.embed_item.forward(variables=item_num,
                                                                cat_variables=item_cat[:, 1].unsqueeze(-1))

        # outer product
        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
        interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))

        # cnn
        feature_map = self.cnn(interaction_map)  # output: batch_size * 16 * 1 * 1
        feature_vec = feature_map.view((-1, self.channel_size))

        # fc
        prediction = self.fc1(torch.cat((feature_vec, user_embeddings, item_embeddings), dim=1))
        prediction = self.fc2(prediction).view((-1))

        if return_weights:
            return prediction, user_weights, item_weights
        else:
            return prediction


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        # init
        self.weight.data.fill_(1.0)

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


class GRN(Module):

    def __init__(self, d_input, d_hidden, dropout, d_output=None):
        super(GRN, self).__init__()
        if d_output is None:
            d_output = d_hidden
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(d_input, d_output, bias=True)
        self.linear_elu = nn.Linear(d_input, d_hidden, bias=True)
        self.linear_pre_glu = nn.Linear(d_hidden, d_hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_post_glu = nn.Linear(d_hidden, 2 * d_output, bias=True)
        self.layer_norm = LayerNorm(d_output, eps=1e-6)

    def forward(self, alpha, context=None):

        if context is not None:
            together = torch.cat((alpha, context), dim=-1)
        else:
            together = alpha
        post_elu = F.elu(self.linear_elu(together))
        pre_glu = self.dropout(self.linear_pre_glu(post_elu))
        return self.layer_norm(F.glu(self.linear_post_glu(pre_glu)) + self.skip(alpha))


# optional dropout and Gated Liner Unit followed by add and norm
class AddNorm(Module):
    def __init__(self, d_hidden, dropout):
        super(AddNorm, self).__init__()
        self.linear_glu = nn.Linear(d_hidden, 2 * d_hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_hidden, eps=1e-6)

    def forward(self, x, residual):
        post_glu = F.glu(self.linear_glu(self.dropout(x)))
        return self.layer_norm(post_glu + residual)


class VSN(Module):
    def __init__(self, d_hidden: int, n_vars: int, cat_vars: int, cat_dims: List[int], dropout: float = 0.0):
        super(VSN, self).__init__()
        self.d_hidden = d_hidden
        self.n_vars = n_vars  # number of numerical features
        self.cat_vars = cat_vars  # number of categorical features
        self.num_embeds = nn.ModuleList(
            nn.Linear(
                in_features=1,
                out_features=self.d_hidden,
            ) for _ in range(self.n_vars)
        )
        self.cat_embeds = nn.ModuleList(
            nn.Embedding(
                num_embeddings=cat_dims[i],
                embedding_dim=self.d_hidden,
            ) for i in range(self.cat_vars)
        )
        self.weight_network = GRN(
            d_input=self.d_hidden * (self.n_vars + self.cat_vars),
            d_hidden=self.d_hidden,
            dropout=dropout,
            d_output=self.n_vars + self.cat_vars,
        )
        self.variable_network = nn.ModuleList(
            GRN(
                d_input=self.d_hidden,
                d_hidden=self.d_hidden,
                dropout=dropout,
            ) for _ in range(self.n_vars + self.cat_vars)
        )

    def forward(self, variables: Tensor, cat_variables: Tensor) -> Tuple[Tensor, Tensor]:
        if variables.shape[-1] != self.n_vars:
            raise ValueError(f'Expected {self.n_vars} numerical variables, but {variables.shape[-1]} given.')
        if cat_variables.shape[-1] != self.cat_vars:
            raise ValueError(f'Expected {self.cat_vars} categorical variables, but {cat_variables.shape[-1]} given.')

        num_embeds = [self.num_embeds[i](variables[:, i:i+1]) for i in range(self.n_vars)]
        cat_embeds = [self.cat_embeds[i](cat_variables[:, i:i+1].squeeze(1)) for i in range(self.cat_vars)]
        all_embeds = num_embeds + cat_embeds
        flatten = torch.cat(all_embeds, dim=-1)  # [B, d_hidden * (n_vars + cat_vars)]
        weight = self.weight_network(flatten).unsqueeze(dim=-2)  # [B, 1, n_vars + cat_vars]
        weight = torch.softmax(weight, dim=-1)
        # [B, d_hidden, n_vars + cat_vars]
        var_encodings = torch.stack(
            tensors=[net(v) for v, net in zip(all_embeds, self.variable_network)],
            dim=-1
        )
        var_encodings = torch.sum(var_encodings * weight, dim=-1)
        return var_encodings, weight
