import argparse
import pickle
from pathlib import Path

import numpy as np

N_TRAIN, N_QUERY, N_BASE = 3000, 1000, 6000
MATRICES = {"dtw": ("DTW", "dis_matrix_dtw_10k.pkl"), "dfd": ("DFD", "dis_matrix_dfrec_10k.pkl"),
            "haus": ("Haus", "dis_matrix_haus_10k.pkl")}


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=4)


def main():
    parser = argparse.ArgumentParser(description="Split the T-Drive full distance matrices.")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()

    query_end = N_TRAIN + N_QUERY
    base_end = query_end + N_BASE
    with open(args.input_root / "traj_list.pkl", "rb") as handle:
        trajectories = pickle.load(handle)
    if len(trajectories) < base_end:
        raise ValueError("traj_list.pkl has {} trajectories, need {}".format(len(trajectories), base_end))
    dump(list(trajectories[:N_TRAIN]), args.output_root / "train_list")
    dump(list(trajectories[N_TRAIN:query_end]), args.output_root / "query_list")
    dump(list(trajectories[query_end:base_end]), args.output_root / "base_list")

    for measure, (dir_name, file_name) in MATRICES.items():
        with open(args.input_root / file_name, "rb") as handle:
            full = np.asarray(pickle.load(handle))
        train = np.array(full[:N_TRAIN, :N_TRAIN], dtype=np.float32)
        test = np.array(full[N_TRAIN:query_end, query_end:base_end], dtype=np.float32)
        del full
        dump(train, args.output_root / dir_name / "{}_train_distance_matrix_result".format(measure))
        dump(test, args.output_root / dir_name / "{}_test_distance_matrix_result".format(measure))
        print("{}: train {} test {}".format(measure, train.shape, test.shape))


if __name__ == "__main__":
    main()
