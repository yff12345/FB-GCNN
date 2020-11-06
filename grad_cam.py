import torch
import torch.nn.functional as F
import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):  # input size: (100, 62, 5)
        self.gradients = []
        x = self.feature_extractor(x)  # output size: (100, 32, 31, 5)
        x.register_hook(self.save_gradient)
        return x


class ModelOutputs:
    def __init__(self, model, feature_extractor, fc):
        self.model = model
        self.fc = fc
        self.feature_extractor = FeatureExtractor(feature_extractor)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        output = self.feature_extractor(x)  # output size : (100, 32, 31, 5)
        batch_size, filter_num, node_num, feature_num = output.size()
        output = torch.reshape(output, [batch_size, node_num * feature_num * filter_num])  # (100, 4960)
        logits = self.fc(output)  # 全连接层...emmmmmm logits size: (100, 7)
        return logits   # (100, 7)


class GradCam:
    def __init__(self, model, feature_extractor, fc):
        self.model = model.to(DEVICE)
        self.flag = model.training
        self.model.eval()  # Sets the module in evaluation mode, dropout and batchnorm are disabled in the evaluation mode

        self.extractor = ModelOutputs(self.model, feature_extractor, fc)

    def forward(self, input_x):
        return self.model(input_x)

    def __call__(self, input_x, index):
        output = self.extractor(input_x.to(DEVICE))     # (100, 7)

        # if output.dim() == 1:
        #     output = output.unsqueeze(0)
        if index is None:
            index = torch.argmax(output, dim=-1)

        index = index.type(torch.long)
        one_hot = torch.zeros_like(output)  # (100, 7)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot.to(DEVICE) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]
        grads_val = F.relu(grads_val)   # (100, 32, 31, 5)
        # weights = grads_val.mean(-1)    # (100, 32, 31)
        weights = grads_val             # (100, 32, 31, 5)

        if self.flag:
            self.model.train()

        return weights.clone().detach()
