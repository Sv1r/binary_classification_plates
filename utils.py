import os
import tqdm
import torch
import shutil
import zipfile
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import settings

sns.set_style('darkgrid')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_preparation(data_folder):
    """Unzip data and split it into train/valid folders"""
    with zipfile.ZipFile(f'{data_folder}/plates.zip', 'r') as zip_object:
        zip_object.extractall(f'{data_folder}')

    train_dir = f'{data_folder}/train'
    valid_dir = f'{data_folder}/valid'

    class_names = ['cleaned', 'dirty']

    for dir_name in [train_dir, valid_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)
    # Move each 6th image to valid folder
    for class_name in class_names:
        source_dir = os.path.join(f'{data_folder}/plates/', 'train', class_name)
        for i, file_name in enumerate(os.listdir(source_dir)):
            if i % 6 != 0:
                current_dir = os.path.join(train_dir, class_name)
            else:
                current_dir = os.path.join(valid_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(current_dir, file_name))

    return train_dir, valid_dir


def show_image_classification(dataloader):
    """Plot 6 images with applied augmentation and labels: {1} - dirty; {0} - cleaned"""
    mean = np.array(settings.MEAN)
    std = np.array(settings.STD)

    x_batch, y_batch = next(iter(dataloader))

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    for x_item, y_item, i, j in zip(x_batch, y_batch, [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]):
        image = x_item.permute(1, 2, 0).numpy()
        image = std * image + mean

        ax[i, j].imshow(image.clip(0, 1))
        ax[i, j].set_title(y_item.item())
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

    fig.tight_layout()
    plt.show()


def train_model(model, model_name, train_dataloader, valid_dataloader, loss, optimizer, num_epochs, scheduler=None):
    """Model training function"""
    print(f'Model {model_name} training')
    train_loss_history, valid_loss_history = [], []
    train_accuracy_history, valid_accuracy_history = [], []
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode
            running_loss = 0.
            running_acc = 0.
            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward and backward
                with torch.set_grad_enabled(phase == 'Train'):
                    predict = model(inputs)
                    loss_value = loss(predict, labels)
                    predict_class = predict.argmax(dim=1)
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss_value.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                # statistics
                running_loss += loss_value.item()
                running_acc += (predict_class == labels.data).float().mean().item()
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            # Loss history
            if phase == 'Train':
                train_loss_history.append(epoch_loss)
                train_accuracy_history.append(epoch_acc)
            else:
                valid_loss_history.append(epoch_loss)
                valid_accuracy_history.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model, train_loss_history, valid_loss_history, train_accuracy_history, valid_accuracy_history


def result_plot(data, models_name):
    """Plot results of models training"""
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    legend_names = ['Train', 'Valid']
    for model in models_name:
        for stage in legend_names:
            loss = data[f'{model}_{stage}_loss'].tolist()
            accuracy = data[f'{model}_{stage}_accuracy'].tolist()
            ax[0].plot(loss, label=f'{model} {stage}')
            ax[1].plot(accuracy, label=f'{model} {stage}')

    for i, j in enumerate(['Loss', 'Accuracy']):
        ax[i].set_title(f'{j} Plot', fontsize=14)
        ax[i].set_xlabel('Epoch', fontsize=12)
        ax[i].set_ylabel(f'{j} Value', fontsize=12)
        ax[i].legend()

    fig.suptitle('Result of Model Training', fontsize=18)
    plt.show()
