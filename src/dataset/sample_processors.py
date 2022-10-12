from importlib.resources import path
from pathlib import Path

import webdataset as wds
import torch

from torchvision import transforms


from .preprocessing import get_preprocessing_pipeline
from .utils import get_dataset_paths, generate_seqs

import matplotlib.pyplot as plt


def rnn_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    # Crop the image
    if config['crop']:
        crop_size = 256 - (2 * config['crop_image_resize'][1])
        images = torch.stack(combined_data['jpeg'], dim=0)[:, :, crop_size:, :]

        # Update image resize shape
        config['image_resize'] = [
            1,
            config['crop_image_resize'][1],
            config['crop_image_resize'][2],
        ]

    else:
        images = torch.stack(combined_data['jpeg'], dim=0)

    # Preprocessing
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Convert the sequence to input and output
    input_seq = images[0:-1, :, :, :]
    output_seq = images[1:, :, :, :]

    return input_seq, output_seq


def semseg_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Get the segmentation labels
    semseg_labels = combined_data['json'][0]['semseg']
    semseg_labels = torch.tensor(semseg_labels).reshape(config['image_size'][1:]).long()
    semseg_labels = transforms.Resize(
        size=(config['image_resize'][1], config['image_resize'][2])
    )(semseg_labels[None, None, ...])

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]
        semseg_labels = semseg_labels[:, :, :crop_size, :]

    images = images[0, :, :, :]
    semseg_labels = semseg_labels[0, 0, :, :]

    return images, semseg_labels


def one_image_samples(samples, config):
    combined_data = {
        k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
    }

    images = torch.stack(combined_data['jpeg'], dim=0)
    preproc = get_preprocessing_pipeline(config)
    images = preproc(images)

    # Crop the image
    if config['crop']:
        crop_size = config['image_resize'][1] - config['crop_image_resize'][1]
        images = images[:, :, :crop_size, :]

    # Convert the sequence to input and output
    input_seq = images[0, :, :, :]
    output_seq = images[0, :, :, :]

    return input_seq, output_seq
