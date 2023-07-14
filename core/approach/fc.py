import os

import numpy
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
from sklearn.model_selection import KFold
from torch import nn
import time

from torch.utils.data import DataLoader

from core.approach.softcup_dataset import SoftCupDataset
from utils.logger import get_logger
import copy


def get_device(gpu, logger):
    if gpu and torch.cuda.is_available():
        logger.info("Using GPU...")
        return torch.device("cuda")
    logger.info("Using CPU...")
    return torch.device("cpu")


class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, linear_sizes, dropout=0.5):
        super(FullyConnected, self).__init__()
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i-1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU(),nn.Dropout(dropout)]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): #[batch_size, in_dim]
        return self.net(x)


class FC:
    def __init__(self, args):
        self.patience = 5
        self.logger = get_logger(args.log_dir, args.model)
        self.fc_params = {
            'in_dim': args.in_dim,
            'out_dim': args.out_dim,
            'linear_sizes': args.hidden,
            'dropout': args.dropout
        }
        self.logger.info('fc params: {}'.format(self.fc_params))
        self.device = get_device(gpu=True, logger=self.logger)
        self.model = FullyConnected(**self.fc_params)
        self.model_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.get_cls = nn.Softmax(dim=-1)
        self.lr = args.learning_rate
        self.epoches = args.epochs
        self.model_save_dir = os.path.join(args.model_dir)

    def evaluate(self, test_loader, datatype="Test"):
        # 开启模型评估模式，锁定droupout和BC层
        self.model.eval()
        abnormal_pred = {}

        y_true, y_pred = [], []
        with torch.no_grad():
            for _, data, label in test_loader:
                model_logits = self.model(data)
                # pred_labels = np.argmax(model_logits.detach().cpu().numpy(),axis=1)
                pred_labels = np.argmax(self.get_cls(model_logits.detach()).cpu().numpy(), axis=1)
                y_true.extend(label.tolist())
                y_pred.extend([int(pred_label) for pred_label in pred_labels])
                for label_i, label_v in enumerate([int(pred_label) for pred_label in pred_labels]):
                    if label_v!=label.tolist()[label_i]:
                        key = "{}-{}".format(label.tolist()[label_i], label_v)
                        if key not in abnormal_pred.keys():
                            abnormal_pred[key] = [data[label_i]]
                        else:
                            abnormal_pred[key].append(data[label_i])
            f1 = f1_score(y_true, y_pred, average='macro')
            pre = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            print(classification_report(y_true,y_pred))
            self.logger.debug("F1, Precison, Recall: {:.3%} {:.3%} {:.3%} F1:{} Precision:{} Recall:{}".format(f1, pre, rec, f1_score(y_true,y_pred,average=None), precision_score(y_true,y_pred,average=None),recall_score(y_true,y_pred,average=None) ))
        return f1

    # 模型运行入口，输入训练数据和测试数据
    def fit_k(self, data, test_data=None, batch_size=32):
        best_hr1, coverage, best_state, eval_res = -1, None, None, None  # evaluation
        pre_loss, worse_count = float("inf"), 0  # early break

        # 优化器，保存当前的参数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)

        # 定义交叉验证的折数和评估指标
        n_splits = 5
        metrics = []

        # 使用 K 折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True)

        for fold_,(train_index, valid_index) in enumerate(kf.split(data)):
            self.model = FullyConnected(**self.fc_params)
            train_df = data.iloc[train_index]
            valid_df = data.iloc[valid_index]
            train_dataset = SoftCupDataset(train_df)
            valid_dataset = SoftCupDataset(valid_df)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            self.fit(train_loader,valid_loader)
            if test_data is not None:
                self.logger.info('test result start')
                self.evaluate(DataLoader(dataset=SoftCupDataset(test_data),batch_size=batch_size,shuffle=False))
                self.logger.info('test result end')


    # 模型运行入口，输入训练数据和测试数据
    def fit(self, train_loader, test_loader=None, evaluation_epoch=5):
        best_hr1, coverage, best_state, eval_res = -1, None, None, None  # evaluation
        pre_loss, worse_count = float("inf"), 0  # early break

        # 优化器，保存当前的参数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)

        # 开始训练
        for epoch in range(1, self.epoches + 1):
            # 启用 Batch Normalization 和 Dropout
            self.model.train()
            # batch-cnt记录当前epoch，处于第几batch,epoch_loss是每batch的累加量
            epoch_time_start = time.time()
            batch_cnt, epoch_loss = 0, 0.0
            y_true, y_pred = [], []
            for _, data, label in train_loader:
                optimizer.zero_grad()
                model_logits = self.model.forward(data)
                loss = self.model_criterion(model_logits, label)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                epoch_loss += loss.item()
                batch_cnt += 1
                pred_labels = np.argmax(self.get_cls(model_logits.detach()).cpu().numpy(),axis=1)
                # pred_labels = np.argmax(model_logits.detach().cpu().numpy(), axis=1)
                y_true.extend(label.tolist())
                y_pred.extend([int(pred_label) for pred_label in pred_labels])
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            f1 = f1_score(y_true, y_pred, average='macro')
            pre = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            print("Epoch {}/{}, training loss: {:.5f} [{:.2f}s], F1, Precison, Recall: {:.3%} {:.3%} {:.3%}".format(epoch, self.epoches, epoch_loss,epoch_time_elapsed, f1, pre, rec))
            print('f1:{}'.format(f1_score(y_true, y_pred, average=None)))
            # self.logger.debug("F1, Precison, Recall: {:.3%} {:.3%} {:.3%}".format(f1, pre, rec))

            ####### early break #######
            # 也就是优化后的(下一轮epoch的)lossi比(前一轮epoch的)lossj大的次数太多，超出patient,就结束训练
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    self.logger.info("Early stop at epoch: {}".format(epoch))
                    break
            else:
                worse_count = 0
                pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch + 1) % evaluation_epoch == 0:
                test_f1 = self.evaluate(test_loader, datatype="Test")
                # 保存f1最大的模型
                if test_f1 > best_hr1:
                    best_hr1 = test_f1
                    best_state = copy.deepcopy(self.model.state_dict())
                    coverage = epoch
                self.save_model(best_state)

        # 训练次数大于5轮的最优模型
        if coverage > 5:
            self.logger.info("* Best result got at epoch {} with f1: {:.4f}".format(coverage, best_hr1))
        else:
            self.logger.info("Unable to convergence!")

        return eval_res, coverage

    def load_model(self, model_save_dir=""):
        self.model.load_state_dict(torch.load(os.path.join(model_save_dir, "fc"), map_location=self.device))

    def save_model(self, state, file=None):
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if file is None: file = os.path.join(self.model_save_dir, "fc")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)



