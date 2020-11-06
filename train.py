import json
import argparse
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chebshev_gcnn import FineGrainedGCNN
from EEGDataset import EEGDataset

from utils import train_utils, model_utils

manual_seed = 2020
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):

    def __init__(self, args, people_num):
        super(Trainer, self).__init__()
        self.args = args
        self.people = people_num

        dataset = EEGDataset
        adj_matrix = dataset.build_graph()

        self.train_dataset = dataset(split=True, people=self.people)
        self.test_dataset = dataset(split=False, people=self.people)

        self.train_loader = DataLoader(self.train_dataset, num_workers=2, batch_size=args.batch_size, shuffle=True,
                                       collate_fn=dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, num_workers=2, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=dataset.collate_fn)

        classes_num = self.train_dataset.classes_num

        self.model = FineGrainedGCNN(adj_matrix, classes_num, args).to(DEVICE)
        self.model.apply(model_utils.weight_init)
        self.model.to(DEVICE)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.early_stopping = train_utils.EarlyStopping(patience=10)
        self.mean_accuracy = train_utils.MeanAccuracy(classes_num)
        self.mean_loss = train_utils.MeanLoss(args.batch_size)

        # print('---' * 20)
        # print('Model architecture:')
        # print('===' * 20)
        # total_parameters = 0
        # for name, param in self.model.named_parameters():
        #     total_parameters += param.nelement()
        #     print('{:15}\t{:25}\t{:5}'.format(name, str(param.shape), param.nelement()))
        # print('===' * 20)
        # print('Total parameters: {}'.format(total_parameters))
        # print('---' * 20)

    def run(self):
        max_acc = 0
        for epoch in range(self.args.max_epochs):
            mloss = self.train()
            acc = self.test(epoch)
            is_best, is_terminate = self.early_stopping(acc)

            if is_terminate:
                break
            if is_best:
                state_dict = self.model.state_dict()
            if acc > max_acc:
                max_acc = acc

            self.lr_scheduler.step(mloss)

        self.model.load_state_dict(state_dict)
        print(
            f'*********************************** 第 {self.people + 1} 个人的 Max Accuracy: {max_acc * 100:.2f}% ***********************************\n\n\n')
        return max_acc

    def train(self):
        self.model.train()
        self.mean_loss.reset()
        desc = "TRAINING - loss: {:.4f}"
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=desc.format(0))
        for step, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            logits = self.model(data, labels)
            loss = self.criterion(logits, labels)
            self.mean_loss.update(loss.cpu().detach().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.desc = desc.format(loss)
            pbar.update(1)
        pbar.close()
        return self.mean_loss.compute()

    def test(self, epoch):
        self.model.eval()
        # self.mean_loss.reset()
        self.mean_accuracy.reset()
        pbar = tqdm(total=len(self.test_loader), leave=False, desc="TESTING")
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits = self.model(data, None)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
                pbar.update(1)
        pbar.close()
        acc = self.mean_accuracy.compute()
        tqdm.write(f"Test Results - Epoch: {epoch} Accuracy: {acc * 100:.2f}%")
        return acc

    def predict(self):
        self.model.eval()

        pre_data = self.test_dataset.eeg_data[:100].to(DEVICE)
        pre_label = self.test_dataset.eeg_labels[:100].to(DEVICE)
        test_output = self.model(pre_data)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        print(pred_y, 'prediction number')
        print(pre_label.cpu().numpy(), 'real number')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gc_layers', type=int, default=1, choices=[1, 2])
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    config = json.load(open('config.json'))['eeg']
    arch_cfg = config['arch']['chebyshev']['layer1']
    graph_cfg = config['graph']
    args = vars(args)
    args.update(arch_cfg)
    args.update(graph_cfg)
    args = Namespace(**args)

    acc_list = []

    for i in range(30):
        trainer = Trainer(args, people_num=i)
        max_acc = trainer.run()
        acc_list.append(max_acc)

    print(f'\n\n\n平均 Accuracy:【{np.mean(acc_list) * 100:.2f}%】\n\n\n')