import torch
import random
import numpy as np
import pandas as pd

import models
import utils
import dataset
import settings

random.seed(settings.RANDOM_STATE)
np.random.seed(settings.RANDOM_STATE)
torch.manual_seed(settings.RANDOM_STATE)
torch.cuda.manual_seed(settings.RANDOM_STATE)


def main():
    df_results = pd.DataFrame()
    loss = torch.nn.CrossEntropyLoss()
    for i, model_name in enumerate(models.models_dict.keys()):
        model = models.models_dict[model_name]
        if i != (len(models.models_dict) - 1):
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = None

        model, train_loss, valid_loss, train_accuracy, valid_accuracy = utils.train_model(model=model,
                                                                                          model_name=model_name,
                                                                                          train_dataloader=dataset.train_dataloader,
                                                                                          valid_dataloader=dataset.valid_dataloader,
                                                                                          loss=loss,
                                                                                          optimizer=optimizer,
                                                                                          num_epochs=settings.EPOCHS,
                                                                                          scheduler=scheduler)
        df_results[f'{model_name}_train_loss'] = train_loss
        df_results[f'{model_name}_valid_loss'] = valid_loss
        df_results[f'{model_name}_train_accuracy'] = train_accuracy
        df_results[f'{model_name}_valid_accuracy'] = valid_accuracy


if __name__ == '__main__':
    main()
