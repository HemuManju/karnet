from torch import nn
import pytorch_lightning as pl
import torch

import torch.nn.functional as F

from src.visualization.visualize import interactive_show_grid

from .layer_config import (
    layers_encoder_256_128,
    layers_decoder_256_128,
    layers_encoder_256_64,
    layers_decoder_256_64,
    layers_encoder_256_256,
    layers_decoder_256_256,
)
from .utils import build_conv_model, build_deconv_model, Flatten, get_model


class ConvNet(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNet, self).__init__()

        # Parameters
        seq_length = hparams['seq_length']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, seq_length, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(seq_length, 16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        x = self.cnn_base(x)
        x = torch.flatten(x, start_dim=1)
        q_values = self.fc(x)
        return q_values


class ConvNetRawSegment(pl.LightningModule):
    def __init__(self, hparams):
        super(ConvNetRawSegment, self).__init__()

        # Parameters
        seq_length = hparams['seq_length']
        n_actions = hparams['n_actions']

        self.example_input_array = torch.randn((1, seq_length, 256, 256))

        # Architecture
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(seq_length, 32, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 48),
            nn.ReLU(),
            nn.Linear(48, n_actions),
        )

    def forward(self, x, x_seg):
        out_1 = self.cnn_base(x)
        out_2 = self.cnn_base(x_seg)
        out_1 = torch.flatten(out_1, start_dim=1)
        out_2 = torch.flatten(out_2, start_dim=1)
        x = out_1 + out_2
        q_values = self.fc(x)
        return q_values


class VAE(pl.LightningModule):
    """
    Simple auto-encoder with MLP network
    Args:
        seq_length: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        self.hidden_size: size of hidden layers
    """

    def __init__(self, hparams, latent_size: int = 128):
        super(VAE, self).__init__()

        # Parameters
        image_size = hparams['image_size']
        self.example_input_array = torch.randn((2, *image_size))

        self.encoder = build_conv_model(hparams['encoder_config'])
        self.decoder = build_deconv_model(hparams['decoder_config'])

        self.fc_mu = nn.Linear(128, latent_size)
        self.fc_log_sigma = nn.Linear(128, latent_size)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma

    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        reconstructed = self.decoder(x)
        return reconstructed

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sigma = log_sigma.exp()
        eps = torch.rand_like(sigma)
        z = eps.mul(sigma).add_(mu)
        reconst = self.decode(z)
        return reconst, mu, log_sigma


class CNNAutoEncoder(pl.LightningModule):
    """
    Simple auto-encoder with MLP network
    Args:
        seq_length: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        self.hidden_size: size of hidden layers
    """

    def __init__(self, hparams):
        super(CNNAutoEncoder, self).__init__()

        # Parameters
        # Crop the image
        if hparams['crop']:
            # Update image resize shape
            hparams['image_resize'] = [
                1,
                hparams['crop_image_resize'][1],
                hparams['crop_image_resize'][2],
            ]
        image_size = hparams['image_resize']

        self.example_input_array = torch.randn((2, *image_size))
        latent_size = hparams['latent_size']

        if latent_size == 64:
            hparams["autoencoder_config"] = {
                "layers_encoder": layers_encoder_256_64,
                "layers_decoder": layers_decoder_256_64,
            }
        elif latent_size == 128:
            hparams["autoencoder_config"] = {
                "layers_encoder": layers_encoder_256_128,
                "layers_decoder": layers_decoder_256_128,
            }
        elif latent_size == 256:
            hparams["autoencoder_config"] = {
                "layers_encoder": layers_encoder_256_256,
                "layers_decoder": layers_decoder_256_256,
            }

        # Encoder and decoder network
        self.encoder = get_model(hparams["autoencoder_config"]['layers_encoder'])
        self.decoder = get_model(hparams["autoencoder_config"]['layers_decoder'])

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def decode(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        reconstructed = self.decoder(x)
        return reconstructed

    def forward(self, x):
        embedding = self.encode(x)
        reconstructed = self.decode(embedding)
        return reconstructed, embedding


class CARNet(pl.LightningModule):
    """
    Simple auto-encoder with MLP network
    Args:
        seq_length: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        self.hidden_size: size of hidden layers
    """

    def __init__(self, hparams, cnn_autoencoder):
        super(CARNet, self).__init__()

        # Parameters
        image_size = hparams['image_resize']
        self.example_input_array = torch.randn((2, 4, *image_size))

        # Encoder and decoder
        self.cnn_autoencoder = cnn_autoencoder

        # RNN
        rnn_input_size = hparams['rnn_input_size']
        hidden_size = hparams['hidden_size']
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(64, 128)

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        # Encoder
        embeddings = self.cnn_autoencoder.encode(
            x.view(batch_size * timesteps, C, H, W)
        )

        # RNN logic
        r_in = embeddings.view(batch_size, timesteps, -1)
        r_out, hidden = self.rnn(r_in)
        rnn_embeddings = r_out.contiguous()

        # print(r_out.shape)
        # rnn_embeddings = self.linear(r_out[:, -1, :])

        # Decoder
        reconstructed = self.cnn_autoencoder.decode(
            rnn_embeddings.view(batch_size * timesteps, -1)
        )
        reconstructed = reconstructed.view(batch_size, timesteps, C, H, W)

        return reconstructed, rnn_embeddings


class BaseConvNet(pl.LightningModule):
    def __init__(self, obs_size):
        super(BaseConvNet, self).__init__()

        # Architecture
        self.flatten = Flatten()
        self.cnn_base = nn.Sequential(  # input shape (4, 256, 256)
            nn.Conv2d(obs_size, 32, kernel_size=7, stride=2),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(),
            # nn.BatchNorm2d(512),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
        )

    def forward(self, x):
        x = self.cnn_base(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class BranchNet(pl.LightningModule):
    def __init__(self, output_size, dropout):
        super(BranchNet, self).__init__()
        self.output_size = output_size
        self.layer_size = 256
        self.dropout = dropout

        self.right_turn = nn.Sequential(
            nn.LazyLinear(self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, output_size),
            nn.ReLU(),
        )

        self.left_turn = nn.Sequential(
            nn.LazyLinear(self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, output_size),
            nn.ReLU(),
        )

        self.straight = nn.Sequential(
            nn.LazyLinear(self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, output_size),
            nn.ReLU(),
        )

        self.lane_follow = nn.Sequential(
            nn.LazyLinear(self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, output_size),
            nn.ReLU(),
        )

        self.branch = nn.ModuleList(
            [self.right_turn, self.left_turn, self.straight, self.lane_follow]
        )

    def forward(self, x, command):
        out = torch.cat(
            [self.branch[i - 1](x_in) for x_in, i in zip(x, command.to(torch.int))]
        )
        return out.view(-1, self.output_size)


class CIRLBasePolicy(pl.LightningModule):
    """A simple convolution neural network"""

    def __init__(self, model_config):
        super(CIRLBasePolicy, self).__init__()

        # Parameters
        self.cfg = model_config
        image_size = self.cfg['image_resize']
        obs_size = self.cfg['obs_size']
        n_actions = self.cfg['n_actions']
        dropout = self.cfg['DROP_OUT']

        # Example inputs
        self.example_input_array = torch.randn(
            (5, obs_size, image_size[1], image_size[2])
        )
        self.example_command = torch.tensor([1, 0, 2, 3, 1])

        self.back_bone_net = BaseConvNet(obs_size)
        self.action_net = BranchNet(output_size=n_actions, dropout=dropout)

    def forward(self, x, command):
        # Testing
        # interactive_show_grid(x[0])
        embedding = self.back_bone_net(x)
        actions = self.action_net(embedding, command)
        return actions


class CIRLCARNet(pl.LightningModule):
    """A simple convolution neural network"""

    def __init__(self, model_config):
        super(CIRLCARNet, self).__init__()

        # Parameters
        self.cfg = model_config
        image_size = self.cfg['image_resize']
        obs_size = self.cfg['obs_size']
        n_actions = self.cfg['n_actions']
        dropout = self.cfg['DROP_OUT']

        # Example inputs
        self.example_input_array = torch.randn((2, obs_size, *image_size[1:]))
        self.example_command = torch.tensor([1, 0, 2, 3, 1])

        self.back_bone_net = BaseConvNet(obs_size)
        self.action_net = BranchNet(output_size=n_actions, dropout=dropout)

        # Future latent vector prediction
        self.carnet = self.set_parameter_requires_grad(self.cfg['carnet'])

    def set_parameter_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, x, command):
        # Future latent vector prediction
        self.carnet.eval()
        reconstructed, rnn_embeddings = self.carnet(x[:, :, None, :, :])

        # Basepolicy
        embedding = self.back_bone_net(x)

        # Combine the embeddings
        combined_embeddings = torch.hstack((rnn_embeddings[:, -1, :], embedding))

        actions = self.action_net(combined_embeddings, command)
        return actions
