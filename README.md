# Cleaned vs Dirty

Original Kaggle [Competition](https://www.kaggle.com/competitions/platesv2/data). This project is about binary classification, but it can be easily updated for multilabel classification task. 

As we do not have many train samples, 'strong' augmentation was applied. ![batch](https://github.com/Sv1r/binary_classification_plates/blob/main/images/train_images.png) Final plots with Cross Entropy Loss function and Accuracy demonstrate, that [ViT](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16) model has better performance on validation images, than two other approaches.

![plot](https://github.com/Sv1r/binary_classification_plates/blob/main/images/training_results.png)
