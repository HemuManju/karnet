import os
import glob

from ast import literal_eval

from collections import deque
import natsort
from subprocess import Popen
import atexit

import numpy as np
import pandas as pd
import deepdish as dd
from skimage.io import imread_collection

import cv2

import webdataset as wds
import jsonpickle

from bagpy import bagreader

from PIL import Image as im

from .utils import get_nonexistant_shard_path


def start_shell_command_and_wait(command):
    p = Popen(command, shell=True, preexec_fn=os.setsid)

    def cleanup():
        os.killpg(os.getpgid(p.pid), 15)

    atexit.register(cleanup)
    p.wait()
    atexit.unregister(cleanup)


def compress_data(config):
    data = {}
    # for log in config['logs'][0]:
    log = 'Log1'
    # for camera in config['camera'][0]:
    camera = 'FL'
    read_path = (
        config['data_dir']
        + 'raw'
        + '/'
        + log
        + '/'
        + camera
        + '_resized_224_bw'
        + '/*.png'
    )
    temp_data = imread_collection(read_path)
    all_images = [image[np.newaxis, ...] for image in temp_data]
    data['test'] = np.concatenate(all_images, dtype=np.int8)
    dd.io.save('test.h5', data)
    return None


def download_CORL2017_dataset():
    google_drive_download_id = "1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY"
    filename_to_save = "./CORL2017ImitationLearningData.tar.gz"
    download_command = (
        'wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm='
        '$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies '
        '--no-check-certificate \"https://docs.google.com/uc?export=download&id={}\" -O- | '
        'sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={}" -O {} && rm -rf /tmp/cookies.txt'.format(
            google_drive_download_id, google_drive_download_id, filename_to_save
        )
    )

    print(download_command)

    # start downloading and wait for it to finish
    start_shell_command_and_wait(download_command)


def create_regression_data(dataset):
    dt = 1.0 / 20.0

    dataset = dataset.unbatched()
    theta_near = []
    theta_far = []
    theta_middle = []
    theta_history = deque([0, 0, 0, 0], maxlen=4)
    steering = []
    integrate_theta = []

    for i, data in enumerate(dataset):
        d = data[2]

        theta_history.append(d[0].numpy())
        theta_near.append(d[0].numpy())
        theta_middle.append(d[1].numpy())
        theta_far.append(d[2].numpy())
        steering.append(d[3].numpy())
        integrate_theta.append(sum(theta_history) * dt)

        if i > 8000:
            break

    # Convert list to numpy and stack
    features = np.vstack(
        (
            np.array(theta_far),
            np.array(theta_middle),
            np.array(theta_near),
            np.array(integrate_theta),
        )
    ).T

    print(features.shape)

    return np.array(steering), features


def decompress_udacity_data(config, file_name):
    read_path = config['real_data_path'] + f'raw/{file_name}.bag'
    bag = bagreader(read_path)
    topics = [
        '/vehicle/steering_report',
        '/vehicle/throttle_report',
        '/vehicle/brake_report',
        '/imu/data',
        '/fix',
        '/center_camera/image_color/compressed',
        '/time_reference',
        '/vehicle/filtered_accel',
        '/vehicle/gps/vel',
    ]

    for topic in topics:
        bag.message_by_topic(topic)

    return None


def read_udacity_data(config):
    # file_name = 'el_camino_south'
    file_name = 'CH03_002'
    read_path = config['real_data_path'] + 'raw/' + file_name

    # Read only if the directory is not already present
    if not os.path.isdir(read_path):
        decompress_udacity_data(config, file_name)

    # Combine throttle, steer, and brake
    topics = {
        '/vehicle-steering_report': ['Time', 'steering_wheel_angle'],
        '/vehicle-throttle_report': ['Time', 'pedal_output'],
        '/vehicle-brake_report': ['Time', 'pedal_output'],
        '/fix': ['Time', 'latitude', 'longitude', 'altitude'],
        '/imu-data': [
            'Time',
            'orientation.x',
            'orientation.y',
            'orientation.z',
            'orientation.w',
            'angular_velocity.x',
            'angular_velocity.y',
            'angular_velocity.z',
            'linear_acceleration.x',
            'linear_acceleration.y',
            'linear_acceleration.z',
        ],
        '/vehicle-gps-vel': ['Time', 'twist.linear.x', 'twist.linear.y'],
    }

    # Read the image data
    chunk_id = 0
    for left in pd.read_csv(
        read_path + '/center_camera-image_color-compressed.csv', chunksize=6250
    ):
        temp = left[['Time', 'data']]
        # Time synchronize the data using pandas 'merge_asof'
        for topic_path, columns in topics.items():
            right = pd.read_csv(read_path + topic_path + '.csv')[columns]
            temp = pd.merge_asof(temp, right, on='Time', direction="nearest")

        temp.to_csv(
            f'/home/hemanth/Desktop/real-data/raw/{file_name}/{file_name}_{chunk_id}.csv'
        )
        chunk_id += 1
    return None


def convert_to_webdataset(config):
    read_path = config['real_data_path']
    path_to_file = read_path + 'processed/real_data' + '_%06d.tar'
    write_path, shard_start = get_nonexistant_shard_path(path_to_file)

    # Create sink
    sink = wds.ShardWriter(
        write_path, maxcount=6250, compress=True, start_shard=shard_start
    )

    # find all the files
    base_name = 'CH03_002'

    files = natsort.natsorted(
        glob.glob(f'/home/hemanth/Desktop/real-data/raw/{base_name}/{base_name}_*.csv')
    )
    index = 0
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for f in files:
        # Read the pandas dataframe
        df = pd.read_csv(f)

        for _, row in df.iterrows():

            t = literal_eval(row['data'])
            buf = np.ndarray(shape=(1, len(t)), dtype=np.uint8, buffer=t)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            image_data = im.fromarray(cv_image).resize(size=(256, 256))

            data = {
                'velocity': [
                    float(row['twist.linear.x']),
                    float(row['twist.linear.y']),
                ],
                'gnss': [
                    float(row['latitude']),
                    float(row['longitude']),
                    float(row['altitude']),
                ],
                'quaternion': [
                    float(row['orientation.x']),
                    float(row['orientation.y']),
                    float(row['orientation.z']),
                    float(row['orientation.w']),
                ],
                'imu': [
                    float(row['linear_acceleration.x']),
                    float(row['linear_acceleration.y']),
                    float(row['linear_acceleration.z']),
                    float(row['angular_velocity.x']),
                    float(row['angular_velocity.y']),
                    float(row['angular_velocity.z']),
                ],
                'steering': float(row['steering_wheel_angle']),
                'throttle': float(row['pedal_output_x']),
                'brake': float(row['pedal_output_y']),
            }

            d = {
                "__key__": f"sample_{index}",
                'jpeg': image_data,
                'json': jsonpickle.encode(data),
            }
            sink.write(d)
            index += 1
    sink.close()


def read_boreas_gps_imu_data(config):
    read_path = config['real_data_path'] + 'sunny/'

    gps = pd.read_csv(read_path + 'applanix/gps_post_process.csv')
    path_to_file = read_path + 'processed/real_data' + '_%06d.tar'
    write_path = get_nonexistant_path(path_to_file)

    sink = wds.ShardWriter(write_path, maxcount=191908, compress=True)

    for index, row in gps.iterrows():
        data = {
            'velocity': [float(row['vel_east']), float(row['vel_north'])],
            'gnss': [
                float(row['latitude']),
                float(row['longitude']),
                float(row['altitude']),
            ],
            'yaw': float(row['heading']),
            'location': [float(row['easting']), float(row['northing']), 0.0],
            'imu': [
                float(row['accelz']),
                float(row['accely']),
                float(row['accelx']),
                float(row['angvel_z']),
                float(row['angvel_y']),
                float(row['angvel_x']),
            ],
        }

        d = {"__key__": "sample%06d" % index, 'json': jsonpickle.encode(data)}
        sink.write(d)
    sink.close()

