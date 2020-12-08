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

# npy_path = 'E:/PycharmProjects/EXPERT_GCN/DATASET/'
npy_path = '/media/data/hanyiik/gcn/dataset/'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = '/media/data/hanyiik/FB_GCN(final)/res/'


class Trainer(object):

    def __init__(self, args, people_num):
        super(Trainer, self).__init__()
        self.args = args
        self.people = people_num
        self.file_path = FILE_PATH

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
            if acc > max_acc:
                max_acc = acc
                torch.save(self.model.state_dict(), f'{self.people + 1}_params.pkl')

            self.lr_scheduler.step(mloss)

        # self.model.load_state_dict(state_dict)
        str_write = f'第 {self.people + 1} 个人的 Max Accuracy: {max_acc * 100:.2f}%'
        print('***********************************' + str_write + '***********************************\n\n\n')
        self.write_result(str_write)
        return max_acc

    def train(self):
        self.model.train()
        self.mean_loss.reset()
        desc = "TRAINING - loss: {:.4f}"
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=desc.format(0))
        for step, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            logits, cam_1, cam_2 = self.model(data, labels)
            loss = self.criterion(logits, labels)
            self.mean_loss.update(loss.cpu().detach().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.desc = desc.format(loss)
            pbar.update(1)
        pbar.close()
        return self.mean_loss.compute()

    def write_result(self, wtr):
        file_name = self.args.file_name
        file_path = self.file_path
        f = open(file_path + file_name, 'a')
        f.write(wtr)
        f.write('\n')
        f.close()

    def test(self, epoch):
        self.model.eval()
        # self.mean_loss.reset()
        self.mean_accuracy.reset()
        pbar = tqdm(total=len(self.test_loader), leave=False, desc="TESTING")
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1, cam_2 = self.model(data, None)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
                pbar.update(1)
        pbar.close()
        acc = self.mean_accuracy.compute()
        tqdm.write(f"Test Results - Epoch: {epoch} Accuracy: {acc * 100:.2f}%")
        return acc

    def visualization(self, people, num):
        self.model.load_state_dict(torch.load(f'{people}_params.pkl'))
        input = np.load(npy_path + 'data_' + 'small' + '/test_dataset_{}.npy'.format(people))[num]
        label = np.load(py_path + 'data_' + 'small' + '/test_labelset_{}.npy'.format(people))[num]

        input = torch.from_numpy(input).unsqueeze(0).to(DEVICE)
        prediction, cam_1, cam_2 = self.model(input)
        prediction = F.softmax(prediction, dim=-1).cpu().detach().numpy()
        pred_y = np.argmax(prediction, axis=1)
        if pred_y == label:
            print('Correct!')
            return cam_1, cam_2
        else:
            print('Error!')
            return pred_y, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gc_layers', type=int, default=1, choices=[1, 2])
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--file_name', type=str, default='k=5、kernel=32、epoch=100.txt')

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