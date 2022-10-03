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

from src.dataset import autoencoder_dataset, imitation_dataset, rnn_dataset
from src.dataset.utils import WebDatasetReader

from src.architectures.nets import (
    CARNet,
    CNNAutoEncoder,
    CIRLCARNet,
    CIRLBasePolicy,
    CIRLRegressorPolicy,
)


from src.models.regression import least_squares
from src.models.imitation import Imitation, Autoencoder
from src.models.utils import load_checkpoint, number_parameters
from src.evaluate.agents import CILAgent, PIThetaNeaFarAgent
from src.evaluate.experiments import CORL2017

from benchmark.run_benchmark import Benchmarking
from benchmark.summary import summarize


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
    # net(net.example_input_array)

    # Dataloader
    data_loader = autoencoder_dataset.webdataset_data_iterator(cfg)
    model = Autoencoder(cfg, net, data_loader)

    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
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
    read_path = f'logs/2022-08-29/AUTOENCODER/autoencoder.ckpt'
    net = CNNAutoEncoder(cfg)
    net = load_checkpoint(net, checkpoint_path=read_path)
    net.eval()
    # net(net.example_input_array)

    # Dataloader
    data_loader = autoencoder_dataset.webdataset_data_iterator(cfg)

    for x, y in data_loader['training']:
        net.eval()
        reconstructured, embeddings = net(x)
        input_test = x[0].detach().cpu().numpy()
        input_test = np.rot90(np.swapaxes(input_test, 2, 0))
        plt.imshow(input_test, origin='lower')
        plt.show()

        input_test = y[0].detach().cpu().numpy()
        input_test = np.rot90(np.swapaxes(input_test, 2, 0))
        plt.imshow(input_test, origin='lower')
        plt.show()

        test = reconstructured[0].detach().cpu().numpy()
        test = np.rot90(np.swapaxes(test, 2, 0))
        plt.imshow(test, origin='lower')
        plt.show()

        break

with skip_run('skip', 'carnet_training') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

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
    # cnn_autoencoder(cnn_autoencoder.example_input_array)

    net = CARNet(cfg, cnn_autoencoder)
    # net(net.example_input_array)

    # Dataloader
    data_loader = rnn_dataset.webdataset_data_iterator(cfg)
    model = Autoencoder(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)

with skip_run('skip', 'imitation_with_basenet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

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

    # Load the network
    net = CIRLBasePolicy(cfg)
    # output = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    model = Imitation(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)

with skip_run('skip', 'imitation_with_basenet_gru') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

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

    # Load the network
    net = CIRLRegressorPolicy(cfg)
    # output = net(net.example_input_array, net.example_command)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    model = Imitation(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)

with skip_run('skip', 'basenet_gru_validation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Load the network
    restore_config = {
        'checkpoint_path': f'logs/2022-09-28/IMITATION/imitation_{navigation_type}.ckpt'
    }
    model = Imitation.load_from_checkpoint(
        restore_config['checkpoint_path'],
        hparams=cfg,
        net=CIRLRegressorPolicy(cfg),
        data_loader=None,
    )
    model.eval()

    # Load the dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    iterator = iter(data_loader['training'])
    for i in range(1000):
        images, commands, actions = next(iterator)
        out = model(images, commands)
        actions = actions.reshape(-1, 2).detach().numpy()
        out = out.reshape(-1, 2).detach().numpy()

        plt.scatter(actions[:, 0], actions[:, 1], c='k')
        plt.scatter(out[:, 0], out[:, 1], c='b')
        plt.xlim([-12, 12])
        plt.pause(0.1)
        plt.cla()

with skip_run('skip', 'regression_for_PI_controller') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    dataset = imitation_dataset.webdataset_data_iterator(cfg)
    least_squares(dataset)

with skip_run('skip', 'regression_for_PI_controller') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/imitation.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/IMITATION'

    # Navigation type
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    dataset = imitation_dataset.webdataset_data_iterator(cfg)
    for data in dataset['training']:
        test = data[2].reshape(128, 2, 5).numpy()
        for d in test:
            plt.scatter(d[0, :], d[1, :])
            plt.pause(0.000000001)
            plt.cla()

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
        file_path='/home/hemanth/Desktop/carla_data_new/Town01_NAVIGATION/one_curve/Town01_HardRainNoon_cautious_000002.tar',
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
        projected_ego = imitation_dataset.project_to_ego_frame(waypoints, data)
        projected_world = imitation_dataset.project_to_world(projected_ego, data)
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
    plt.scatter(reproj_test[:, 0], reproj_test[:, 1], marker='o')
    plt.show()

with skip_run('skip', 'imitation_with_carnet') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

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
    read_path = f'logs/2022-07-07/IMITATION/imitation_{navigation_type}.ckpt'
    cnn_autoencoder = CNNAutoEncoder(cfg)
    carnet = CARNet(cfg, cnn_autoencoder)
    carnet = load_checkpoint(carnet, checkpoint_path=read_path)
    cfg['carnet'] = carnet

    # Testing
    # reconstructed, rnn_embeddings = carnet(carnet.example_input_array)

    net = CIRLCARNet(cfg)
    # net(net.example_input_array, net.example_command)

    # net(net.example_input_array)

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)
    model = Imitation(cfg, net, data_loader)
    if cfg['check_point_path'] is None:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
    else:
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=cfg['NUM_EPOCHS'],
            logger=logger,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=cfg['check_point_path'],
            enable_progress_bar=False,
        )
    trainer.fit(model)

with skip_run('skip', 'theta_estimation') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
    cfg['logs_path'] = cfg['logs_path'] + str(date.today()) + '/WARMSTART'

    # Random seed
    gpus = get_num_gpus()
    torch.manual_seed(cfg['pytorch_seed'])

    # Checkpoint
    navigation_type = cfg['navigation_types'][0]
    cfg['raw_data_path'] = cfg['raw_data_path'] + f'/{navigation_type}'

    # Dataloader
    data_loader = imitation_dataset.webdataset_data_iterator(cfg)

    t_near = []
    t_far = []
    t_middle = []
    for i, data in enumerate(data_loader['training']):
        t_near.append(data[2][:, 0].numpy().tolist())
        t_middle.append(data[2][:, 1].numpy().tolist())
        t_far.append(data[2][:, 2].numpy().tolist())
        if i > 100:
            break

    t_near = sum(t_near, [])
    t_middle = sum(t_middle, [])
    t_far = sum(t_far, [])

    plt.hist(np.array(t_near), bins=10)
    plt.hist(np.array(t_middle), bins=10)
    plt.hist(np.array(t_far), bins=10)

    # plt.plot(np.array(t_far), 'o')
    # plt.plot(np.array(t_middle), 's')
    # plt.plot(np.array(t_near), '.')

    print(np.max(np.array(t_far)))
    print(np.min(np.array(t_far)))
    plt.show()

with skip_run('skip', 'benchmark_trained_model') as check, check():
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
            'checkpoint_path': f'logs/2022-08-25/WARMSTART/imitation_{navigation_type}.ckpt'
        }

        model = Imitation.load_from_checkpoint(
            restore_config['checkpoint_path'],
            hparams=cfg,
            net=CIRLBasePolicy(cfg),
            data_loader=None,
        )

        # Change agent
        agent = CILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)
        # agent = PIThetaNeaFarAgent(model, cfg)

        # Run the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)
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
        agent = CILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)
        # agent = PIDCILAgent(model, cfg)

        # Run the benchmark
        benchmark = Benchmarking(core, cfg, agent, experiment_suite)
        benchmark.run(config, exp_id)

    # Kill all servers
    kill_all_servers()

with skip_run('skip', 'summarize_benchmark') as check, check():
    # Load the configuration
    cfg = yaml.load(open('configs/warmstart.yaml'), Loader=yaml.SafeLoader)
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
        path = f'logs/benchmark_results/{town}_{navigation_type}_{weather}_0/measurements.csv'
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
    data_loader = rnn_dataset.webdataset_data_iterator(cfg)

    for data in data_loader['training']:
        output = model(data[0][0:1], data[1][0:1])
        print(data[2][0:1])
        # print(torch.max(data[2][:, 0] / 20))
        print(output)
        print('-------------------')
