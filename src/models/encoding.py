import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_msssim import MS_SSIM

from .utils import ChamferDistance, calc_ssim_kernel_size


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super(Autoencoder, self).__init__()
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)

        k = calc_ssim_kernel_size(self.h_params['image_resize'], levels=5)
        criterion = MS_SSIM(win_size=k, data_range=1, size_average=True, channel=1)
        criterion_l1 = nn.L1Loss()

        loss = 1.0 - criterion(output, y) + criterion_l1(output, y)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Predict and calculate loss
        output, embedding = self.forward(x)

        k = calc_ssim_kernel_size(self.h_params['image_resize'], levels=5)
        criterion = MS_SSIM(win_size=k, data_range=1, size_average=True, channel=1)
        criterion_l1 = nn.L1Loss()

        loss = 1.0 - criterion(output, y) + criterion_l1(output, y)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.data_loader['training']

    def val_dataloader(self):
        return self.data_loader['validation']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.h_params['LEARNING_RATE'])
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=5, factor=0.95, verbose=True
        )

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'losses/val_loss',
        }

        if self.h_params['use_scheduler']:
            return [optimizer], [scheduler]
        else:
            return [optimizer]


class SemanticSegmentation(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.squeeze(1), labels)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.squeeze(1), labels)

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss


class RNNSegmentation(Autoencoder):
    def __init__(self, hparams, net, data_loader):
        super().__init__(hparams, net, data_loader)
        self.h_params = hparams
        self.net = net
        self.data_loader = data_loader

        # Save hyperparameters
        self.save_hyperparameters(self.h_params)

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)

        self.log('losses/train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch

        # Predict and calculate loss
        output, embeddings = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)  # Target shape: [batch, seq length, H, W]

        self.log('losses/val_loss', loss, on_step=False, on_epoch=True)
        return loss
