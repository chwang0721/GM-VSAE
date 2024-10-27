import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']


def height2lat(height):
    return height / 110.574


def width2lng(width):
    return width / 111.320 / 0.99974


def devide(trajs):
    print('----------Dividing dataset----------')
    sd_cnt = defaultdict(list)

    for traj in trajs:
        s, d = traj[0], traj[-1]
        sd_cnt[(s, d)].append(traj)

    train_data = []
    test_data = []

    for trajs in sd_cnt.values():
        if len(trajs) >= min_sd_traj_num:
            train_trajs, test_trajs = trajs[:-test_traj_num], trajs[-test_traj_num:]
            for traj in train_trajs:
                train_data.append(traj)

            for traj in test_trajs:
                test_data.append(traj)

    print('Train trajectory num:', len(train_data))
    print('Test trajectory num: ', len(test_data))
    return train_data, test_data


def main():
    lat_size, lng_size = height2lat(grid_height), width2lng(grid_width)
    lat_grid_num = int((boundary['max_lat'] - boundary['min_lat']) / lat_size) + 1
    lng_grid_num = int((boundary['max_lng'] - boundary['min_lng']) / lng_size) + 1
    print('Grid size:', (lat_grid_num, lng_grid_num))
    print('----------Preprocessing----------')

    trajectories = pd.read_csv(f'{data_dir}/{data_name}.csv', usecols=['POLYLINE'])
    trajs = []
    for _, traj in tqdm(trajectories.iterrows(), total=len(trajectories)):
        traj_seq = []
        valid = True
        polyline = json.loads(traj['POLYLINE'])
        for lng, lat in polyline:
            if in_boundary(lat, lng, boundary):
                grid_i = (lat - boundary['min_lat']) // lat_size
                grid_j = (lng - boundary['min_lng']) // lng_size
                traj_seq.append(int(grid_i * lng_grid_num + grid_j))
            else:
                valid = False
                break
        if valid:
            if shortest <= len(traj_seq) <= longest:
                trajs.append(traj_seq)

    train_data, test_data = devide(trajs)
    np.save('data/porto/train_data.npy', np.array(train_data, dtype=object))
    np.save('data/porto/test_data.npy', np.array(test_data, dtype=object))
    print('Fnished!')


if __name__ == '__main__':
    data_dir = '../datasets/porto'
    data_name = 'porto'
    grid_height, grid_width = 0.1, 0.1
    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}
    min_sd_traj_num = 25
    test_traj_num = 5
    shortest, longest = 20, 1200
    main()
