import torch
import random
import numpy as np
import pandas as pd

import utils
import models
import dataset
import settings

# Fix random behavior
random.seed(settings.RANDOM_STATE)
np.random.seed(settings.RANDOM_STATE)
torch.manual_seed(settings.RANDOM_STATE)
torch.cuda.manual_seed(settings.RANDOM_STATE)


def main():
    # Init results dataframe with loss and accuracy history
    df = pd.DataFrame()
    loss = torch.nn.CrossEntropyLoss()
    for i, model_name in enumerate(models.models_dict.keys()):
        model = models.models_dict[model_name]
        # For Vit model we have to use specific optimization algorithm
        if i == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)
        if i == 1:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=.064, momentum=.00001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.973)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = None

        model, train_loss, valid_loss, train_accuracy, valid_accuracy = utils.train_model(
            model=model,
            model_name=model_name,
            train_dataloader=dataset.train_dataloader,
            valid_dataloader=dataset.valid_dataloader,
            loss=loss,
            optimizer=optimizer,
            num_epochs=settings.EPOCHS,
            scheduler=scheduler
        )
        # Add results for each model
        df[f'{model_name}_Train_loss'] = train_loss
        df[f'{model_name}_Valid_loss'] = valid_loss
        df[f'{model_name}_Train_accuracy'] = train_accuracy
        df[f'{model_name}_Valid_accuracy'] = valid_accuracy
    utils.result_plot(df, list(models.models_dict.keys()))


if __name__ == '__main__':
    main()
