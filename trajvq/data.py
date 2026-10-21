import pickle
from pathlib import Path

import numpy as np

GRID_SIZE = 0.002
MEASURE_DIRS = {"dtw": "DTW", "haus": "Haus", "dfd": "DFD"}


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_dataset(root, measure):
    root = Path(root)
    splits = [list(_load_pickle(root / name)) for name in ("train_list", "query_list", "base_list")]
    sizes = tuple(len(split) for split in splits)
    distance_dir = root / MEASURE_DIRS[measure]
    train_dist = np.asarray(_load_pickle(distance_dir / "{}_train_distance_matrix_result".format(measure)))
    test_dist = np.asarray(_load_pickle(distance_dir / "{}_test_distance_matrix_result".format(measure)))
    for name, matrix, expected in (("train", train_dist, sizes[:1] * 2), ("test", test_dist, sizes[1:])):
        if matrix.shape != expected:
            raise ValueError("{} distances have shape {}, expected {}".format(name, matrix.shape, expected))
        if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
            raise ValueError("{} distances must be finite floating-point values".format(name))
    return splits[0] + splits[1] + splits[2], sizes, train_dist, test_dist


def _grid_ids(values_per_traj, grid_size):
    min_value = np.min([np.min(values) for values in values_per_traj])
    max_value = np.max([np.max(values) for values in values_per_traj])
    ids = [[int((value - min_value) / grid_size) for value in values] for values in values_per_traj]
    return ids, int((max_value - min_value) / grid_size) + 1, min_value


def to_grid(trajectories, grid_size=GRID_SIZE):
    lons = [[point[0] for point in traj] for traj in trajectories]
    lats = [[point[1] for point in traj] for traj in trajectories]
    lon_ids, lon_size, lon_min = _grid_ids(lons, grid_size)
    lat_ids, lat_size, lat_min = _grid_ids(lats, grid_size)
    lon_coords = [np.asarray([(point[0] - lon_min) * 100 for point in traj], dtype=np.float32) for traj in trajectories]
    lat_coords = [np.asarray([(point[1] - lat_min) * 100 for point in traj], dtype=np.float32) for traj in trajectories]
    return lon_ids, lat_ids, (lon_size, lat_size), lon_coords, lat_coords


def pad_sequences(sequences, max_length):
    padded = np.zeros((len(sequences), max_length, 1), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence), 0] = sequence
    return padded
