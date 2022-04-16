import os

import numpy as np
import pandas as pd

from cv2 import imread
import wandb

import torch
from torch.utils.data import Dataset
from albumentations import Resize, Compose
from albumentations.augmentations.transforms import Normalize

from config import Config
from tqdm import tqdm


# Data

class WoodDataset(Dataset):
    def __init__(self, create_train=True) -> None:
        self.transform = Compose([Resize(Config.SIZE, Config.SIZE), Normalize()])
        self.create_train = create_train

        if create_train:
            self.data = pd.read_csv('train.csv')
            self.data = self.data.sort_values('id').reset_index().drop('index', axis=1)
            self.data['class'] = self.data['class'].astype(int)
        else:
            self.data = pd.read_csv('sample_submission.csv')
            self.data['id'] = self.data['id'].astype(str)

    def __getitem__(self, index) -> dict:

        id_ = self.data.iloc[index]['id']
        if self.create_train:
            image = imread(f'Data/Train/{id_}.png').astype(np.float32)
            processed_image = self.transform(image=image)
            return np.moveaxis(processed_image['image'], 2, 0), torch.tensor(self.data.iloc[index]['class'],
                                                                             dtype=torch.int64)
        else:
            image = imread(f'Data/Test/{id_}.png').astype(np.float32)
            processed_image = self.transform(image=image)
            return np.moveaxis(processed_image['image'], 2, 0)

    def __len__(self) -> None:
        return self.data.shape[0]


def train_one_epoch(model, train_dataloader, criterion, optimizer, device=Config.DEVICE):
    rolling_loss = 0
    for images, labels in tqdm(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        y_pred = model.forward(images)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        rolling_loss += loss.detach().cpu().numpy()

    rolling_loss /= (len(train_dataloader) * Config.BATCH_SIZE)
    out_dict = {"Train loss": rolling_loss}

    return out_dict


def _validate(model, val_dataloader, val_metric=None, val_criterion=None, device=Config.DEVICE):
    if val_metric and val_criterion:
        rolling_loss = 0
        rolling_metric = 0
        for features, true_predictions in tqdm(val_dataloader):
            features = features.to(device)
            true_predictions = true_predictions.to(device)

            predictions = model.forward(features)
            loss = val_criterion(predictions, true_predictions)
            metric = val_metric(predictions, true_predictions)

            rolling_loss += loss.detach().cpu().numpy()
            rolling_metric += metric

        rolling_loss /= (len(val_dataloader) * Config.BATCH_SIZE)
        rolling_metric /= (len(val_dataloader) * Config.BATCH_SIZE)

        out_dict = {
            'Validation loss': rolling_loss,
            'Validation metric': rolling_metric
        }

        return out_dict

    elif val_metric:
        rolling_metric = 0
        for features, true_predictions in tqdm(val_dataloader):
            features = features.to(device)
            true_predictions = true_predictions.to(device)

            predictions = model.forward(features)
            metric = val_metric(predictions, true_predictions)

            rolling_metric += metric.detach().cpu().numpy()

        rolling_metric /= (len(val_dataloader) * Config.BATCH_SIZE)

        out_dict = {
            'Validation metric': rolling_metric
        }

        return out_dict

    elif val_criterion:
        rolling_loss = 0
        for features, true_predictions in tqdm(val_dataloader):
            features = features.to(device)
            true_predictions = true_predictions.to(device)

            predictions = model.forward(features)
            loss = val_criterion(predictions, true_predictions)

            rolling_loss += loss.detach().cpu().numpy()

        rolling_loss /= (len(val_dataloader) * Config.BATCH_SIZE)

        out_dict = {
            'Validation loss': rolling_loss
        }

        return out_dict

    else:
        raise AttributeError


def train(model, train_dataloader, criterion, optimizer, val_dataloader=None,
          val_metric=None, val_criterion=None, n_epochs=10, device=Config.DEVICE):
    model.to(device)
    model.train()
    torch.save(model.state_dict(), 'Models/test')
    for i in range(n_epochs):
        loss_dict = train_one_epoch(model=model, train_dataloader=train_dataloader,
                                    criterion=criterion, optimizer=optimizer, device=device)
        print(f"Epoch {i + 1}/{n_epochs} is finished, train loss is {loss_dict['Train loss']}.")

        if val_dataloader:
            validation_dict = _validate(model=model, val_dataloader=val_dataloader, val_metric=val_metric,
                                        val_criterion=val_criterion, device=device)
            loss_dict = {**loss_dict, **validation_dict}
            print(f'Validation for epoch {i + 1}/{n_epochs} is finished.')

        wandb.log(loss_dict)


def predict(model, test_dataloader, device=Config.DEVICE):
    model = model.to(device)
    predicted_labels = torch.tensor([]).to(device)
    for images in tqdm(test_dataloader):
        images = images.to(device)
        y_pred = model.forward(images)
        predicted_labels = torch.cat((predicted_labels, y_pred.argmax(1).detach()), 0)
    return predicted_labels.cpu()


def save_model(model, filename, path='Models'):
    if filename not in os.listdir(path):
        torch.save(model.state_dict(), path + '/' + filename)
        return path + '/' + filename
    else:
        i = 1
        while True:
            if not filename + str(i) in os.listdir(path):
                torch.save(model.state_dict(), path + '/' + filename + str(i))
                break
            i += 1
        return path + '/' + filename + str(i)
