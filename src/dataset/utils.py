import os
import collections
from pathlib import Path
from itertools import islice, cycle, product

import numpy as np

import natsort

from torchvision import transforms


def nested_dict():
    return collections.defaultdict(nested_dict)


def generate_seqs(src, concatenate_samples, nsamples=3, config=None):
    it = iter(src)
    result = tuple(islice(it, nsamples))
    if len(result) == nsamples:
        yield concatenate_samples(result, config)
    for elem in it:
        result = result[1:] + (elem,)
        yield concatenate_samples(result, config)


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)


def get_dataset_paths(config):
    paths = {}
    data_split = config['data_split']
    read_path = config['raw_data_path']
    for key, split in data_split.items():
        combinations = [
            '_'.join(item)
            for item in list(product(split['town'], split['season'], split['behavior']))
        ]

        # Get all the tar files
        temp = [find_tar_files(read_path, combination) for combination in combinations]

        # Concatenate all the paths and assign to dict
        paths[key] = sum(temp, [])  # Not a good way, but it is fun!
    return paths


def run_fast_scandir(dir, ext, logs=None):  # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_image_json_files(read_path):
    # Read image files and sort them
    _, file_list = run_fast_scandir(read_path, [".jpeg"])
    image_files = natsort.natsorted(file_list)

    # Read json files and sort them
    _, file_list = run_fast_scandir(read_path, [".json"])
    json_files = natsort.natsorted(file_list)
    return image_files, json_files


def get_preprocessing_pipeline(config):

    preproc = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(
                size=(config['image_resize'][1], config['image_resize'][2])
            ),
            # transforms.Normalize(mean=[0.5], std=[1.0]),
            # transforms.ToTensor(),
        ]
    )
    return preproc


def rotate(points, origin, angle):
    return (points - origin) * np.exp(complex(0, angle)) + origin
