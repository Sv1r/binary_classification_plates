import torch
import torchvision

import utils
import settings
import transforms

train_dir, valid_dir = utils.data_preparation(data_folder=settings.DATA_FOLDER)

train_dataset = torchvision.datasets.ImageFolder(train_dir, transforms.train_transform)
valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transforms.valid_transform)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    pin_memory=True)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)
if settings.PLOT_BATCH:
    utils.show_image_classification(train_dataloader)
