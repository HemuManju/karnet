import os
from datetime import date
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import webdataset as wbs

import torch
import pytorch_lightning as pl

from benchmark.core.carla_core import CarlaCore
from benchmark.core.carla_core import kill_all_servers

from src.data.create_data import create_regression_data
from src.data.stats import classification_accuracy

from src.dataset.sample_processors import (
    one_image_samples,
    rnn_samples,
    semseg_samples,
    rnn_samples_with_kalman,
)
from src.dataset import imitation_dataset
from src.dataset.utils import (
    WebDatasetReader,
    show_image,
    get_webdataset_data_iterator,
    labels_to_cityscapes_palette,
)

from src.architectures.nets import (
    CARNet,
    CARNetExtended,
    CNNAutoEncoder,
    ResNetAutoencoder,
    CIRLCARNet,
    CIRLBasePolicy,
    ResCARNet,
    AutoRegressorBranchNet,
    CIRLRegressorPolicy,
)


from src.models.imitation import Imitation
from src.models.encoding import (
    Autoencoder,
    SemanticSegmentation,
    RNNSegmentation,
    RNNEncoder,
    KalmanRNNEncoder,
)
from src.models.kalman import ExtendedKalmanFilter
from src.models.utils import load_checkpoint, number_parameters
from src.evaluate.agents import PIDCILAgent, PIDKalmanAgent
from src.evaluate.experiments import CORL2017

from benchmark.run_benchmark import Benchmarking
from benchmark.summary import summarize

from tests.test_loss import test_ssim_loss_function

import yaml
from utils import skip_run, get_num_gpus

with skip_run('skip', 'carnet_autoencoder_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/autoencoder.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/AUTOENCODER'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Add navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'autoencoder',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name=f'autoencoder')

    # Setup
    net = CNNAutoEncoder(cfg)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, one_image_samples)

    if cfg['check_point_path'] is None:
        model = Autoencoder(cfg, net, data_loader)
    else:
        model = Autoencoder.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'carnet_semseg_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/autoencoder.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/SEGMENTATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Add navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'segmentation',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(cfg['logs_path'], name=f'segmentation')

    # Setup
    net = ResNetAutoencoder(cfg)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, semseg_samples)

    if cfg['check_point_path'] is None:
        model = SemanticSegmentation(cfg, net, data_loader)
    else:
        model = SemanticSegmentation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'verify_autoencoder') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/autoencoder.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/AUTOENCODER'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Setup
    read_path = f'logs/2022-11-09/AUTOENCODER/last.ckpt'
    net = CNNAutoEncoder(cfg)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, semseg_samples)
    model = Autoencoder.load_from_checkpoint(
        read_path,
        hparams=cfg,
        net=net,
        data_loader=data_loader,
    )
    model.eval()

    fig, ax = plt.subplots(nrows=1, ncols=2)

    for x, y in data_loader['training']:
        model.eval()
        with torch.no_grad():
            reconstructured, embeddings = model(x)
            show_image(x[0], ax[0])
            show_image(reconstructured[0], ax[1])
            plt.pause(0.1)
            plt.cla()

with skip_run('skip', 'verify_segmentation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/autoencoder.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/AUTOENCODER'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Setup
    read_path = f'logs/2022-10-15/SEGMENTATION/segmentation.ckpt'
    net = ResNetAutoencoder(cfg)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, semseg_samples)
    model = SemanticSegmentation.load_from_checkpoint(
        read_path,
        hparams=cfg,
        net=net,
        data_loader=data_loader,
    )
    model.eval()

    for x, y in data_loader['training']:
        model.eval()
        with torch.no_grad():
            reconstructured, embeddings = net(x)
            show_image(x[0])
            labels = torch.argmax(reconstructured[0], dim=0)
            show_image(labels_to_cityscapes_palette(labels))

with skip_run('skip', 'carnet_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/CARNET'
    cfg['message'] = 'Autoencoder changed 2 filter to 16 filter'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Add navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'carnet_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'carnet_{navigation_type}'
    )

    # Setup
    cnn_autoencoder = CNNAutoEncoder(cfg)
    read_path = 'logs/2022-11-09/AUTOENCODER/last.ckpt'
    cnn_autoencoder = load_checkpoint(cnn_autoencoder, read_path)
    # cnn_autoencoder(cnn_autoencoder.example_input_array)

    net = CARNet(cfg, cnn_autoencoder)
    # net(net.example_input_array)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, rnn_samples)
    if cfg['check_point_path'] is None:
        model = RNNEncoder(cfg, net, data_loader)
    else:
        model = Autoencoder.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'verify_carnet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup
    cnn_autoencoder = CNNAutoEncoder(cfg)
    read_path = 'logs/2022-10-25/AUTOENCODER/last.ckpt'
    cnn_autoencoder = load_checkpoint(cnn_autoencoder, read_path)
    # cnn_autoencoder(cnn_autoencoder.example_input_array)

    model = CARNet(cfg, cnn_autoencoder)
    model.eval()

    data_loader = get_webdataset_data_iterator(cfg, rnn_samples)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    for x, y in data_loader['training']:
        model.eval()
        with torch.no_grad():
            reconstructured, embeddings = model(x)
            show_image(x[0, 0, ...], ax[0])
            show_image(reconstructured[0, 0, ...], ax[1])
            plt.pause(0.1)
            plt.cla()

with skip_run('skip', 'rescarnet_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/RNN_SEGMENTATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Add navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'rnn_segmentation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'rnn_segmentation_{navigation_type}'
    )

    # Setup the networks
    res_autoencoder = ResNetAutoencoder(cfg)
    read_path = 'logs/2022-10-15/SEGMENTATION/segmentation.ckpt'
    res_autoencoder = load_checkpoint(res_autoencoder, read_path)

    net = ResCARNet(cfg, res_autoencoder)
    # net(net.example_input_array)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, rnn_samples)
    model = RNNSegmentation(cfg, net, data_loader)

    if cfg['check_point_path'] is None:
        model = RNNSegmentation(cfg, net, data_loader)
    else:
        model = SemanticSegmentation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'test_loss_function') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/autoencoder.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/AUTOENCODER'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Data loader
    data_loader = get_webdataset_data_iterator(cfg, one_image_samples)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    for x, y in data_loader['training']:
        show_image(x[0, ...], ax[0])
        show_image(x[63, ...], ax[1])
        test_ssim_loss_function(cfg, x[0:1, ...], x[63:64, ...])

with skip_run('skip', 'dataset_analysis') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Dataset reader
    reader = WebDatasetReader(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data_new/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000002.tar',
    )
    dataset = reader.get_dataset(concat_n_samples=1)
    waypoint_data = []
    location = []
    reprojected = []
    direction = []

    for i, data in enumerate(dataset):
        data = data['json'][0]
        waypoints = data['waypoints']
        direction.append(np.array(data['moving_direction']))
        projected_ego = imitation_dataset.project_to_ego_frame(data)
        projected_world = imitation_dataset.project_to_world_frame(projected_ego, data)
        reprojected.append(projected_world)
        waypoint_data.append(waypoints)
        location.append(data['location'])
        if i > 1000:
            break

    test_way = np.array(sum(waypoint_data, []))
    directions = np.array(direction)
    test_loc = np.array(location)
    reproj_test = np.concatenate(reprojected)
    plt.quiver(
        test_loc[:, 0],
        test_loc[:, 1],
        directions[:, 0],
        directions[:, 1],
        linewidths=10,
    )
    plt.scatter(test_way[:, 0], test_way[:, 1])
    # plt.scatter(test_loc[:, 0], test_loc[:, 1], marker='s')
    plt.scatter(reproj_test[:, 0], reproj_test[:, 1], s=10, marker='s')
    plt.show()

with skip_run('skip', 'kalman_analysis') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/KALMAN'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Dataset reader
    reader = WebDatasetReader(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000002.tar',
    )
    dataset = reader.get_dataset(concat_n_samples=1)
    predicted = []
    location = []
    last_location = None
    velocity = []
    velocity_true = []

    # Kalman filter
    ekf = ExtendedKalmanFilter(cfg)

    for i, data in enumerate(dataset):
        data = data['json'][0]
        waypoints = data['waypoints']

        # EKF update step
        corrected, predict = ekf.update(data)

        location.append(data['location'])
        predicted.append(predict[0:2])
        velocity.append(predict[2:4])
        velocity_true.append(data['velocity'])

        if i > 1000:
            break

    location = np.array(location) / 400
    predicted = np.array(predicted)
    velocity = np.array(velocity)
    velocity_true = np.array(velocity_true)

    plt.scatter(location[:, 0], location[:, 1], label='Ground Truth')
    # plt.scatter(test_loc[:, 0], test_loc[:, 1], marker='s')
    plt.scatter(predicted[:, 0], predicted[:, 1], s=10, marker='s', label='Predicted')
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')
    plt.legend()
    plt.show()
    plt.scatter(np.arange(0, velocity.shape[0]), velocity[:, 0], label='Predicted')
    plt.scatter(
        np.arange(0, velocity.shape[0]), velocity_true[:, 0] / 20, label='Ground Truth'
    )
    plt.legend()
    plt.show()

    plt.scatter(np.arange(0, velocity.shape[0]), velocity[:, 1], label='Predicted')
    plt.scatter(
        np.arange(0, velocity.shape[0]), velocity_true[:, 1] / 20, label='Ground Truth'
    )
    plt.legend()
    plt.show()

with skip_run('run', 'carnet_with_kalman_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/CARNET_KALMAN'
    cfg['message'] = 'CARNet training with kalman update within the network'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Add navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Kalman filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'carnet_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'carnet_{navigation_type}'
    )

    # # Setup
    cnn_autoencoder = CNNAutoEncoder(cfg)
    read_path = 'logs/2022-11-09/AUTOENCODER/last.ckpt'
    cnn_autoencoder = load_checkpoint(cnn_autoencoder, read_path)
    # cnn_autoencoder(cnn_autoencoder.example_input_array)

    net = CARNetExtended(cfg, cnn_autoencoder)
    # net(net.example_input_array)

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, rnn_samples_with_kalman)
    if cfg['check_point_path'] is None:
        model = KalmanRNNEncoder(cfg, net, data_loader)
    else:
        model = KalmanRNNEncoder.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'imitation_with_carnet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'
    cfg['message'] = 'Random action net'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup
    # Load the backbone network
    read_path = 'logs/2022-11-14/CARNET/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    # read_path = 'logs/action_net.pt'
    # action_net = load_checkpoint(
    #     action_net, checkpoint_path=read_path, only_weights=True
    # )
    cfg['action_net'] = action_net

    # Testing
    # reconstructed, rnn_embeddings = carnet(carnet.example_input_array)

    net = CIRLCARNet(cfg)
    # waypoint, speed = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'verify_carnet_imitation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Load the network
    restore_config = {
        'checkpoint_path': f'logs/2022-10-26/IMITATION/imitation_{navigation_type}.ckpt'
    }
    #  Load the backbone network
    read_path = 'logs/2022-10-25/CARNET/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    read_path = 'logs/action_net.pt'
    action_net = load_checkpoint(
        action_net, checkpoint_path=read_path, only_weights=True
    )
    cfg['action_net'] = action_net
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CIRLCARNet(cfg),
        data_loader=None,
    )
    # model = CIRLCARNet(cfg)
    model.eval()

    # Load the dataloader
    dataset = imitation_dataset.webdataset_data_test_iterator(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000007.tar',
    )

    predicted_waypoints = []
    true_waypoints = []
    test = []
    speed_pred = []
    speed = []
    for i, data in enumerate(dataset):

        images, commands, actions = data[0], data[1], data[2]
        output = model(images.unsqueeze(0), torch.tensor(commands).unsqueeze(0))
        actions = actions[0].reshape(-1, 2).detach().numpy()
        out = output[0].reshape(-1, 2).detach().numpy()

        # Speed
        speed_pred.append(output[1].detach().numpy())
        speed.append(data[3]['speed'])

        # Waypoints from the data
        test.append(data[3]['waypoints'])

        # Project to the world
        predicted = imitation_dataset.project_to_world_frame(out, data[3])
        predicted_waypoints.append(predicted)

        groud_truth = imitation_dataset.project_to_world_frame(actions, data[3])
        true_waypoints.append(groud_truth)

        if i > 1000:
            break

    plt.plot(np.arange(0, len(speed)), speed_pred)
    plt.plot(np.arange(0, len(speed)), speed)
    plt.show()

    true_waypoints = np.vstack(true_waypoints)
    plt.scatter(true_waypoints[:, 0], true_waypoints[:, 1])

    # test = np.array(sum(test, []))
    # plt.scatter(test[:, 0], test[:, 1], s=10, marker='s')

    pred_waypoints = np.vstack(predicted_waypoints)
    plt.scatter(pred_waypoints[:, 0], pred_waypoints[:, 1])
    plt.show()

with skip_run('skip', 'imitation_with_kalman_carnet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION_KALMAN'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='losses/val_loss',
        dirpath=cfg['logs_path'],
        save_top_k=1,
        filename=f'imitation_{navigation_type}',
        mode='min',
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        cfg['logs_path'], name=f'imitation_{navigation_type}'
    )

    # Setup
    # Load the backbone network
    read_path = 'logs/2022-11-14/CARNET/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    # read_path = 'logs/action_net.pt'
    # action_net = load_checkpoint(
    #     action_net, checkpoint_path=read_path, only_weights=True
    # )
    cfg['action_net'] = action_net

    # Kalmnn filter
    cfg['ekf'] = ExtendedKalmanFilter(cfg)

    # Testing
    # reconstructed, rnn_embeddings = carnet(carnet.example_input_array)

    net = CIRLCARNet(cfg)
    # waypoint, speed = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    if cfg['check_point_path'] is None:
        model = Imitation(cfg, net, data_loader)
    else:
        model = Imitation.load_from_checkpoint(
            cfg['check_point_path'],
            hparams=cfg,
            net=net,
            data_loader=data_loader,
        )
    # Trainer
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=cfg['NUM_EPOCHS'],
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(model)

with skip_run('skip', 'verify_carnet_imitation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Load the network
    restore_config = {
        'checkpoint_path': f'logs/2022-10-26/IMITATION/imitation_{navigation_type}.ckpt'
    }
    #  Load the backbone network
    read_path = 'logs/2022-10-25/CARNET/last.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Action net
    action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
    read_path = 'logs/action_net.pt'
    action_net = load_checkpoint(
        action_net, checkpoint_path=read_path, only_weights=True
    )
    cfg['action_net'] = action_net
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CIRLCARNet(cfg),
        data_loader=None,
    )
    # model = CIRLCARNet(cfg)
    model.eval()

    # Load the dataloader
    dataset = imitation_dataset.webdataset_data_test_iterator(
        cfg,
        file_path=f'/home/hemanth/Desktop/carla_data/Town01_NAVIGATION/{navigation_type}/Town01_HardRainNoon_cautious_000007.tar',
    )

    predicted_waypoints = []
    true_waypoints = []
    test = []
    speed_pred = []
    speed = []
    for i, data in enumerate(dataset):

        images, commands, actions = data[0], data[1], data[2]
        output = model(images.unsqueeze(0), torch.tensor(commands).unsqueeze(0))
        actions = actions[0].reshape(-1, 2).detach().numpy()
        out = output[0].reshape(-1, 2).detach().numpy()

        # Speed
        speed_pred.append(output[1].detach().numpy())
        speed.append(data[3]['speed'])

        # Waypoints from the data
        test.append(data[3]['waypoints'])

        # Project to the world
        predicted = imitation_dataset.project_to_world_frame(out, data[3])
        predicted_waypoints.append(predicted)

        groud_truth = imitation_dataset.project_to_world_frame(actions, data[3])
        true_waypoints.append(groud_truth)

        if i > 1000:
            break

    plt.plot(np.arange(0, len(speed)), speed_pred)
    plt.plot(np.arange(0, len(speed)), speed)
    plt.show()

    true_waypoints = np.vstack(true_waypoints)
    plt.scatter(true_waypoints[:, 0], true_waypoints[:, 1])

    # test = np.array(sum(test, []))
    # plt.scatter(test[:, 0], test[:, 1], s=10, marker='s')

    pred_waypoints = np.vstack(predicted_waypoints)
    plt.scatter(pred_waypoints[:, 0], pred_waypoints[:, 1])
    plt.show()

with skip_run('skip', 'benchmark_trained_imitaion_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    ekf = ExtendedKalmanFilter(cfg)
    experiment_suite = CORL2017(experiment_cfg, ekf)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):
        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-11-05/IMITATION_KALMAN/imitation_{navigation_type}.ckpt'
        }

        read_path = 'logs/2022-10-25/CARNET/last.ckpt'
        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNet(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        # Action net
        action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
        read_path = 'logs/action_net.pt'
        action_net = load_checkpoint(
            action_net, checkpoint_path=read_path, only_weights=True
        )
        cfg['action_net'] = action_net
        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLCARNet(cfg),
            data_loader=None,
        )

        agent = PIDCILAgent(model=model, config=cfg)

        # Setup the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)

        # Run the benchmark
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()(model, cfg)

with skip_run('skip', 'benchmark_trained_imitaion_kalman_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    cfg = yaml.load(open('configs/carnet.yaml'), Loader=yaml.SafeLoader)
    ekf = ExtendedKalmanFilter(cfg)
    experiment_suite = CORL2017(experiment_cfg, ekf)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):
        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-11-05/IMITATION_KALMAN/imitation_{navigation_type}.ckpt'
        }

        read_path = 'logs/2022-10-25/CARNET/last.ckpt'
        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNet(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        # Action net
        action_net = AutoRegressorBranchNet(dropout=0, hparams=cfg)
        read_path = 'logs/action_net.pt'
        action_net = load_checkpoint(
            action_net, checkpoint_path=read_path, only_weights=True
        )
        cfg['action_net'] = action_net
        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLCARNet(cfg),
            data_loader=None,
        )

        agent = PIDKalmanAgent(model=model, config=cfg)

        # Setup the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)

        # Run the benchmark
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()(model, cfg)

with skip_run('skip', 'benchmark_trained_carnet_model') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)

    # Experiment_config and experiment suite
    experiment_cfg = yaml.load(open('configs/experiments.yaml'), Loader=yaml.SafeLoader)
    experiment_suite = CORL2017(experiment_cfg)

    # Carla server
    # Setup carla core and experiment
    kill_all_servers()
    os.environ["CARLA_ROOT"] = cfg['carla_server']['carla_path']
    core = CarlaCore(cfg['carla_server'])

    # Get all the experiment configs
    all_experiment_configs = experiment_suite.get_experiment_configs()
    for exp_id, config in enumerate(all_experiment_configs):

        # Update the summary writer info
        town = config['town']
        navigation_type = config['navigation_type']
        weather = config['weather']
        config['summary_writer']['directory'] = f'{town}_{navigation_type}_{weather}'

        # Update the model
        restore_config = {
            'checkpoint_path': f'logs/2022-08-25/WARMSTART/{navigation_type}_last.ckpt'
        }

        # Setup
        # Load the backbone network
        read_path = f'logs/2022-07-07/IMITATION/imitation_{navigation_type}.ckpt'
        cnn_autoencoder = CNNAutoEncoder(cfg)
        carnet = CARNet(cfg, cnn_autoencoder)
        carnet = load_checkpoint(carnet, checkpoint_path=read_path)
        cfg['carnet'] = carnet

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLCARNet(cfg),
            data_loader=None,
        )

        # Change agent
        # agent = CILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)
        agent = PIDCILAgent(model, cfg)

        # Run the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()

with skip_run('skip', 'summarize_benchmark') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # towns = ['Town02', 'Town01']
    # weathers = ['ClearSunset', 'SoftRainNoon']
    # navigation_types = ['straight', 'one_curve', 'navigation']

    towns = ['Town01']
    weathers = ['SoftRainNoon']  #'ClearSunset',
    navigation_types = ['navigation']

    for town, weather, navigation_type in itertools.product(
        towns, weathers, navigation_types
    ):
        path = f'logs/benchmark_results/{town}_{navigation_type}_{weather}_4/measurements.csv'
        print('-' * 32)
        print(town, weather, navigation_type)
        summarize(path)

with skip_run('skip', 'benchmark_trained_model') as check, check():
    # Load the configuration
    navigation_type = 'one_curve'

    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)

    raw_data_path = cfg['raw_data_path']
    cfg['raw_data_path'] = raw_data_path + f'/{navigation_type}'

    restore_config = {'checkpoint_path': 'logs/2022-06-06/one_curve/warmstart.ckpt'}
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CARNet(cfg),
        data_loader=None,
    )
    model.freeze()
    model.eval()
    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Dataloader
    data_loader = get_webdataset_data_iterator(cfg, rnn_samples)

    for data in data_loader['training']:
        output = model(data[0][0:1], data[1][0:1])
        print(data[2][0:1])
        # print(torch.max(data[2][:, 0] / 20))
        print(output)
        print('-------------------')
