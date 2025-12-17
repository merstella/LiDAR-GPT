import os
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import json

# --- CẤU HÌNH ---
DATA_ROOT = "./data/nuscenes"
ANN_FILE = "./unified_nuscenes_infos.json"
SAMPLE_IDX = 50
N_SWEEPS = 0
N_POINTS = 10000


def get_matrix(pose):
    trans = np.array(pose['translation'])
    rot = Quaternion(pose['rotation']).rotation_matrix
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def load_pc(path):
    full_path = os.path.join(DATA_ROOT, path)
    if not os.path.exists(full_path):
        print(f"⚠️ File not found: {full_path}")
        return np.zeros((0, 4), dtype=np.float32)
    pts = np.fromfile(full_path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]


def visualize_pc(points, name="Point Cloud"):
    if points.shape[0] == 0:
        print(f"⚠️ {name}: empty")
        return

    xyz = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    z = xyz[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.zeros_like(xyz)
    colors[:, 0] = z_norm
    colors[:, 1] = 1 - z_norm
    colors[:, 2] = 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries([pcd, axes], window_name=name)


def run_debug():
    print(f"📂 Loading metadata: {ANN_FILE}")
    with open(ANN_FILE, "r") as f:
        infos = json.load(f)

    info = infos[SAMPLE_IDX]
    print(f"🔍 Sample index: {SAMPLE_IDX}")
    print(f"🆔 Token: {info['sample_token']}")

    # ===============================
    # STEP 1: RAW KEYFRAME
    # ===============================
    print("\n=== STEP 1: RAW KEYFRAME ===")
    ref_points = load_pc(info['lidar_path'])
    print(f"➡️ Raw points: {ref_points.shape[0]}")
    visualize_pc(ref_points, "Step 1: Raw Keyframe")

    # ===============================
    # STEP 2: ACCUMULATE SWEEPS
    # ===============================
    print(f"\n=== STEP 2: ACCUMULATE SWEEPS (n={N_SWEEPS}) ===")

    ref_calib = info['calibrated_sensor']
    ref_pose = info['ego_pose']

    ref_lidar2ego = get_matrix(ref_calib)
    ref_ego2global = get_matrix(ref_pose)

    ref_global2ego = np.linalg.inv(ref_ego2global)
    ref_ego2lidar = np.linalg.inv(ref_lidar2ego)

    global2ref_lidar = ref_ego2lidar @ ref_global2ego

    all_points = [ref_points]

    for i, sweep in enumerate(info.get("sweeps", [])):
        if i >= N_SWEEPS:
            break

        sweep_points = load_pc(sweep['lidar_path'])
        if sweep_points.shape[0] == 0:
            continue

        sweep_calib = sweep['calibrated_sensor']
        sweep_pose = sweep['ego_pose']

        sweep_lidar2ego = get_matrix(sweep_calib)
        sweep_ego2global = get_matrix(sweep_pose)

        sweep_lidar2global = sweep_ego2global @ sweep_lidar2ego
        transform = global2ref_lidar @ sweep_lidar2global

        xyz = sweep_points[:, :3]
        homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_ref = (transform @ homo.T).T[:, :3]

        sweep_points[:, :3] = xyz_ref
        all_points.append(sweep_points)

    accumulated = np.concatenate(all_points, axis=0)
    print(f"➡️ After accumulation: {accumulated.shape[0]} points")
    visualize_pc(accumulated, f"Step 2: Accumulated ({N_SWEEPS} sweeps)")

    # ===============================
    # STEP 3: FILTER RANGE
    # ===============================
    print("\n=== STEP 3: FILTER RANGE ===")

    x_min, x_max = -54, 54
    y_min, y_max = -54, 54
    z_min, z_max = -5, 3

    mask = (
        (accumulated[:, 0] >= x_min) & (accumulated[:, 0] <= x_max) &
        (accumulated[:, 1] >= y_min) & (accumulated[:, 1] <= y_max) &
        (accumulated[:, 2] >= z_min) & (accumulated[:, 2] <= z_max)
    )

    filtered = accumulated[mask]
    print(f"➡️ After range filter: {filtered.shape[0]} points")
    visualize_pc(filtered, "Step 3: Filtered Range")

    # ===============================
    # STEP 4: UNI3D PREPROCESS
    # ===============================
    print("\n=== STEP 4: UNI3D PREPROCESS ===")

    num_pts = filtered.shape[0]
    print(f"➡️ Before sampling: {num_pts} points")

    if num_pts >= N_POINTS:
        idx = np.random.choice(num_pts, N_POINTS, replace=False)
    else:
        idx = np.random.choice(num_pts, N_POINTS, replace=True)

    xyz = filtered[idx, :3]
    print(f"➡️ After sampling: {xyz.shape[0]} points")

    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale

    print(f"➡️ After normalization: {xyz.shape[0]} points (unchanged)")
    visualize_pc(xyz, "Step 4: Uni3D Normalized (Unit Sphere)")

    print("\n✅ Debug pipeline finished successfully.")


if __name__ == "__main__":
    run_debug()
