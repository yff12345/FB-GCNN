import torch
import torch.nn.functional as F
import numpy as np
import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.seterr(divide='ignore',invalid='ignore')


class FeatureExtractor:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor  # ChebshevGCNN

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):  # input size: (100, 62, 5)
        self.gradients = []
        x = self.feature_extractor(x)  # output size: (100, 32, 62, 5)
        x.register_hook(self.save_gradient)
        return x


class ModelOutputs:
    def __init__(self, feature_extractor, fc):
        self.fc = fc
        self.feature_extractor = FeatureExtractor(feature_extractor)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        features = self.feature_extractor(x)  # features size : (100, 32, 62, 5)
        batch_size, filter_num, node_num, feature_num = features.size()
        output = torch.reshape(features, [batch_size, node_num * feature_num * filter_num])  # (100, 9920)
        output = self.fc(output)  # 全连接层...emmmmmm output size: (100, 7)
        return features, output
        # [features]经过一层 ChebshevGCNN 的输出 (100, 32, 62, 5)
        # [output]经过一层 ChebshevGCNN 后再经过一层 fc 的最终输出 (100, 7)


class GradCam:
    def __init__(self, model, feature_extractor, fc, rate):
        self.model = model.to(DEVICE)
        self.flag = model.training
        self.model.eval()  # Sets the module in evaluation mode, dropout and batchnorm are disabled in the evaluation mode
        self.rate = rate

        self.extractor = ModelOutputs(feature_extractor, fc)

    def forward(self, input_x):
        return self.model(input_x)

    def __call__(self, input_x, index):
        features, output = self.extractor(input_x.to(DEVICE))

        if output.dim() == 1:
            output = output.unsqueeze(0)

        if index is None:
            index = torch.argmax(output, dim=-1)

        index = index.type(torch.long)
        one_hot = torch.zeros_like(output)  # (100, 7)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot.to(DEVICE) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]  # (100, 32, 62, 5)
        # grads_val = F.relu(grads_val)

        weight = grads_val.cpu().detach().numpy()   # (100, 32, 62, 5)
        weight = np.mean(weight, axis=(2, 3))       # (100, 32) 每个样本取 32 个特征图所对应的梯度图的均值，每个样本对应 32 个均值。
        target = features.cpu().detach().numpy()    # (100, 32, 62, 5)
        cam = np.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32)  # (100, 62, 5)
        nodes_cam = np.zeros((target.shape[0], target.shape[2]), dtype=np.float32)  # (100, 62)
        new_cam_list = []

        for item in range(cam.shape[0]):
            for i, w in enumerate(weight[item]):
                cam[item] += w * target[item, i, :, :]

            # cam[item] = np.maximum(cam[item], 0)  # 小于 0 的地方置零。。
            # cam[item] = cam[item] - np.min(cam[item])
            # cam[item] = cam[item] / np.max(cam[item])

        # pdb.set_trace()
        for j, one_cam in enumerate(cam):
            mask_sum = one_cam.sum(axis=1)  # (62,)
            mask_max = np.max(mask_sum)
            mask_min = np.min(mask_sum)
            mask_sum = (mask_sum - mask_min)/(mask_max - mask_min)
            nodes_cam[j] = mask_sum

        new_cam = np.sign(np.sign(nodes_cam - self.rate) + 1)
        new_cam = torch.from_numpy(new_cam).to(DEVICE)

        for my_cam in new_cam:
            new_cam_list.append(torch.nonzero(my_cam).squeeze(1))

        if self.flag:
            self.model.train()

        return new_cam_list, nodes_cam
    '''
        :: 
    '''
