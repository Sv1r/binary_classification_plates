import torch
import torchvision

# ResNet18
resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 2)
# MobileNet_v3_small
mobilenet_v3 = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
for param in mobilenet_v3.parameters():
    param.requires_grad = False
mobilenet_v3.classifier[3] = torch.nn.Linear(mobilenet_v3.classifier[3].in_features, 2)
# vit_16
vit_16 = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
for param in vit_16.parameters():
    param.requires_grad = False
vit_16.heads.head = torch.nn.Linear(vit_16.heads.head.in_features, 2)

models_list = [resnet18, mobilenet_v3, vit_16]
models_names = ['ResNet18', 'MobileNet_v3', 'ViT_16']
models_dict = dict(zip(models_names, models_list))
