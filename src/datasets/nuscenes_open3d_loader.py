import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from lidarGPT_datasets import filter_range, uni3d_preprocess


def count_points_in_bin(bin_file: Path) -> int:
    """Quickly count points in a nuScenes .pcd.bin without fully parsing."""
    bin_path = Path(bin_file)
    if not bin_path.is_file():
        raise FileNotFoundError(f"Could not find file: {bin_path}")

    bytes_per_point = 5 * np.dtype(np.float32).itemsize  # x, y, z, intensity, ring
    file_size = bin_path.stat().st_size

    if file_size % bytes_per_point != 0:
        raise ValueError(
            f"File size {file_size} not divisible by {bytes_per_point}; cannot infer point count cleanly."
        )

    return file_size // bytes_per_point


def read_nuscenes_bin(bin_file: Path) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Load a nuScenes lidar .pcd.bin file into an Open3D PointCloud and raw numpy array."""
    bin_path = Path(bin_file)
    if not bin_path.is_file():
        raise FileNotFoundError(f"Could not find file: {bin_path}")

    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"Expected data multiple of 5 floats (x, y, z, intensity, ring) in {bin_path}")

    points = raw.reshape(-1, 5)  # [x, y, z, intensity, ring]
    xyz = points[:, :3]
    intensity = points[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if intensity.size:
        # Map intensity to grayscale for quick inspection.
        norm = intensity / (np.max(intensity) + 1e-8)
        colors = np.repeat(norm[:, None], 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, points


def visualize_point_cloud(pcd: o3d.geometry.PointCloud, window_name: str = "nuScenes LiDAR"):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes], window_name=window_name)


def main():
    parser = argparse.ArgumentParser(description="Load and view a nuScenes LiDAR .pcd.bin file with Open3D.")
    parser.add_argument("bin_path", help="Relative or absolute path to a nuScenes .pcd.bin file.")
    parser.add_argument(
        "--data-root",
        default="data/nuscenes",
        help="Base folder of the nuScenes dataset when bin_path is relative (default: data/nuscenes).",
    )
    parser.add_argument("--no-vis", action="store_true", help="Only load the point cloud without opening a viewer.")
    parser.add_argument("--raw", action="store_true", help="Visualize raw points instead of filtered+Uni3D processed.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of points for Uni3D sampling when preprocessing (default: 10000).",
    )
    args = parser.parse_args()

    bin_file = Path(args.bin_path)
    if not bin_file.is_absolute():
        bin_file = Path(args.data_root) / bin_file

    fast_count = count_points_in_bin(bin_file)
    pcd_raw, raw_points = read_nuscenes_bin(bin_file)
    loaded_count = len(pcd_raw.points)

    print(f"Point count (fast size check): {fast_count}")
    print(f"Loaded {loaded_count} points from {bin_file}")

    if args.raw:
        pcd_to_show = pcd_raw
        print("Visualizing raw point cloud.")
    else:
        filtered = filter_range(raw_points[:, :4])  # keep xyz + intensity
        processed = uni3d_preprocess(filtered, n_points=args.n_points)  # (N, 6) xyz + fake rgb
        print(f"Filtered to {filtered.shape[0]} points; sampled/normalized to {processed.shape[0]} points.")

        pcd_processed = o3d.geometry.PointCloud()
        pcd_processed.points = o3d.utility.Vector3dVector(processed[:, :3])
        pcd_processed.colors = o3d.utility.Vector3dVector(processed[:, 3:6])
        pcd_to_show = pcd_processed
        print("Visualizing filtered + Uni3D-processed point cloud.")

    if not args.no_vis:
        visualize_point_cloud(pcd_to_show)


if __name__ == "__main__":
    main()
