import math

import numpy as np
import webdataset as wds
import torch
import scipy.interpolate

from .utils import get_preprocessing_pipeline, rotate, get_dataset_paths, generate_seqs

import matplotlib.pyplot as plt


def get_projections(x, on):
    # Make the third dimension zero
    on[2] = 0.0
    x[2] = 0.0
    proj = np.zeros((1, 2))
    parallel = np.dot(x, on) / np.dot(on, on) * on
    perpendicular = x - parallel

    proj[0, 0] = -perpendicular[0]  # The directions are reversed in carla
    proj[0, 1] = parallel[1]

    # Find the rotation angle such the movement direction is always positive
    if on[1] < 0:
        theta = math.pi
        R = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        proj = R.dot(proj.T).flatten()
    return proj


def project_waypoints(waypoints, vector):
    projected_waypoints = np.zeros((len(waypoints), 2))
    for i, waypoint in enumerate(waypoints):
        projected_waypoints[i, :] = get_projections(waypoint, vector)
    return projected_waypoints


def post_process_action(data, config):
    if config['action_processing_id'] == 1:
        action = torch.tensor(
            [data['throttle'], (data['steer'] + 1) * 2, data['brake']]
        )
    elif config['action_processing_id'] == 2:
        action = torch.tensor(
            [
                data['speed'] / 20,
                data['throttle'],
                (data['steer'] + 1) * 2,
                data['brake'],
            ]
        )
    elif config['action_processing_id'] == 3:
        action = torch.tensor([data['speed'] / 5.55, (data['steer'] + 1)])
    elif config['action_processing_id'] == 4:
        # Calculate theta near and theta far
        theta_near, theta_middle, theta_far = calculate_theta_near_far(
            data['waypoints'], data['location']
        )
        action = torch.tensor([theta_near, theta_middle, theta_far, data['steer']])

    elif config['action_processing_id'] == 5:
        # Calculate theta near and theta far
        ego_frame_waypoints = calculate_ego_frame_waypoints(data)
        points = ego_frame_waypoints[0:5, :].flatten('F').astype(np.float32)
        action = torch.from_numpy(points)
    else:
        action = torch.tensor([data['throttle'], data['steer'], data['brake']])

    return action


def calculate_ego_frame_waypoints(data):
    # Direction vector
    v_vec = np.array(data['moving_direction'])

    # Origin shift
    shifted_waypoints = np.array(data['waypoints']) - np.array(data['location'])

    # Projected points
    ego_frame_waypoints = project_waypoints(shifted_waypoints, v_vec)

    return ego_frame_waypoints


def calculate_angle(v0, v1):
    theta = np.arctan2(np.cross(v0, v1), np.dot(v0, v1)).astype(np.float32)
    if theta > 3.0:
        theta = 0.0
    return theta


def resample_waypoints(waypoints, current_location, resample=False):
    if resample:
        xy = np.array(waypoints)
        x = xy[:, 0]
        y = xy[:, 1]

        # Add initial location
        x = np.insert(x, 0, current_location[0])
        y = np.insert(y, 0, current_location[1])

        # Shift the frame to origin
        # x = x - x[0]
        # y = y - y[0]

        # get the cumulative distance along the contour
        dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
        dist_along = np.concatenate(([0], dist.cumsum()))

        # build a spline representation of the contour
        try:
            spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)
            # resample it at smaller distance intervals
            interp_d = np.linspace(dist_along[0], dist_along[-1], 10)
            interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)
        except ValueError:
            interp_x = x
            interp_y = y

        processed_waypoints = np.vstack((interp_x, interp_y)).T
    else:
        processed_waypoints = np.array(waypoints)[:, 0:2]
    return processed_waypoints


def calculate_theta_near_far(waypoints, location):
    # NOTE: The angles returned are in radians.
    # The x and y position are given in world co-ordinates, but theta near and theta far should
    # be calculated in the direction of movement.

    if len(waypoints) > 1:
        # resample waypoints
        current_location = np.array(location[0:2])
        waypoints = resample_waypoints(waypoints, current_location)

        # From the vectors taking ego's location as origin
        v0 = waypoints[0] - current_location

        # Select the second point as the near points
        point_select = 1
        v1 = waypoints[point_select] - current_location
        theta_near = calculate_angle(v0, v1)

        point_select = 2
        v1 = waypoints[point_select] - current_location
        theta_middle = calculate_angle(v0, v1)

        # Select the fourth point as the far point
        point_select = 4
        v1 = waypoints[point_select] - current_location
        theta_far = calculate_angle(v0, v1)
    else:
        theta_far, theta_middle, theta_near = 0.0, 0.0, 0.0

    return float(theta_near), float(theta_middle), float(theta_far)


def concatenate_samples(samples, config):
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

    preproc = get_preprocessing_pipeline(config)
    images = preproc(images).squeeze(1)
    last_data = samples[-1]['json']

    if last_data['modified_direction'] in [-1, 5, 6]:
        command = 4
    else:
        command = last_data['modified_direction']

    # Post processing according to the ID
    action = post_process_action(last_data, config)

    return images, command, action


def webdataset_data_iterator(config):
    # Get dataset path(s)
    paths = get_dataset_paths(config)

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    SEQ_LEN = config['obs_size']
    number_workers = config['number_workers']

    # Create train, validation, test datasets and save them in a dictionary
    data_iterator = {}

    for key, path in paths.items():
        if path:
            dataset = (
                wds.WebDataset(path, shardshuffle=False)
                .decode("torchrgb")
                .then(generate_seqs, concatenate_samples, SEQ_LEN, config)
            )
            data_loader = wds.WebLoader(
                dataset,
                num_workers=number_workers,
                shuffle=False,
                batch_size=BATCH_SIZE,
            )
            if key in ['training', 'validation']:
                dataset_size = 6250 * len(path)
                data_loader.length = dataset_size // BATCH_SIZE

            data_iterator[key] = data_loader

    return data_iterator
