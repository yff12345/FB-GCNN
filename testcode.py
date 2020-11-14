import torch.nn as nn
import torch

input = torch.randn(8, 3, 50, 100)
print(input.requires_grad)
# False

net = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
)

for name, param in net.named_parameters():
    print(name, param.shape, param.requires_grad)
# 0.weight True
# 0.bias True
# 1.weight True
# 1.bias True

output = net(input)
print(output.requires_grad)
# True