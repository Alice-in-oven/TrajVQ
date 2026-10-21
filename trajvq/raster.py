import numpy as np
from scipy.ndimage import distance_transform_edt

IMAGE_CHANNELS = {
    "dtw5_no_direction": 5,
    "haus4_no_density_no_turn": 4,
    "dfd6_order_dir": 6,
    "dfd4_order": 4,
}


def _safe_average(sum_map, count_map):
    result = np.zeros_like(sum_map, dtype=np.float32)
    valid = count_map > 0
    result[valid] = sum_map[valid] / count_map[valid]
    return result


def _trajectory_stats(x_ids, y_ids, traj, shape):
    stats = {name: np.zeros(shape, dtype=np.float32) for name in (
        "hit_count", "dx_sum", "dy_sum", "step_count", "progress_sum", "progress_count", "start", "end")}
    n = min(len(x_ids), len(y_ids), len(traj))
    if n == 0:
        return stats

    lon = np.asarray([point[0] for point in traj[:n]], dtype=np.float32)
    lat = np.asarray([point[1] for point in traj[:n]], dtype=np.float32)
    cumulative = np.zeros((n,), dtype=np.float32)
    step_lengths = np.zeros((max(n - 1, 0),), dtype=np.float32)
    for j in range(1, n):
        step = float(np.hypot(float(lon[j] - lon[j - 1]), float(lat[j] - lat[j - 1])))
        step_lengths[j - 1] = step
        cumulative[j] = cumulative[j - 1] + step
    total = float(cumulative[-1])

    for j in range(n):
        cx, cy = int(x_ids[j]), int(y_ids[j])
        stats["hit_count"][cx, cy] += 1.0
        if j == 0:
            stats["start"][cx, cy] = 1.0
        if j == n - 1:
            stats["end"][cx, cy] = 1.0
        stats["progress_count"][cx, cy] += 1.0
        if total > 0.0:
            stats["progress_sum"][cx, cy] += cumulative[j] / total
        elif n > 1:
            stats["progress_sum"][cx, cy] += float(j) / float(n - 1)
        if j == 0:
            continue
        step = float(step_lengths[j - 1])
        if step > 0.0:
            stats["dx_sum"][cx, cy] += float(lon[j] - lon[j - 1]) / step
            stats["dy_sum"][cx, cy] += float(lat[j] - lat[j - 1]) / step
            stats["step_count"][cx, cy] += 1.0
    return stats


def _canvas(stats, image_mode):
    hit = stats["hit_count"]
    occupancy = (hit > 0.0).astype(np.float32)
    progress = np.clip(_safe_average(stats["progress_sum"], stats["progress_count"]), 0.0, 1.0)
    start = np.clip(stats["start"], 0.0, 1.0)
    end = np.clip(stats["end"], 0.0, 1.0)

    if image_mode == "dtw5_no_direction":
        density = np.zeros_like(hit)
        max_hit = float(np.max(hit))
        if max_hit > 0.0:
            density = np.clip(np.log1p(hit) / np.log1p(max_hit), 0.0, 1.0)
        channels = [occupancy, density, progress, start, end]
    elif image_mode == "haus4_no_density_no_turn":
        dist = distance_transform_edt(1.0 - occupancy).astype(np.float32)
        max_dist = float(np.max(dist))
        dist = np.log1p(dist) / np.log1p(max_dist) if max_dist > 0.0 else np.zeros_like(dist, dtype=np.float32)
        channels = [occupancy, np.clip(dist, 0.0, 1.0), start, end]
    elif image_mode == "dfd6_order_dir":
        dx = np.clip(_safe_average(stats["dx_sum"], stats["step_count"]), -1.0, 1.0)
        dy = np.clip(_safe_average(stats["dy_sum"], stats["step_count"]), -1.0, 1.0)
        channels = [occupancy, dx, dy, progress, start, end]
    elif image_mode == "dfd4_order":
        channels = [occupancy, progress, start, end]
    else:
        raise ValueError("Unsupported image_mode: {}".format(image_mode))

    canvas = np.zeros((len(channels),) + hit.shape, dtype=np.float32)
    for i, channel in enumerate(channels):
        canvas[i] = channel
    return canvas


def build_rasters(x_id_lists, y_id_lists, trajectories, shape, image_mode):
    canvases = [_canvas(_trajectory_stats(x_ids, y_ids, traj, shape), image_mode)
                for x_ids, y_ids, traj in zip(x_id_lists, y_id_lists, trajectories)]
    return np.asarray(canvases, dtype=np.float32).reshape((len(canvases), IMAGE_CHANNELS[image_mode]) + tuple(shape))
