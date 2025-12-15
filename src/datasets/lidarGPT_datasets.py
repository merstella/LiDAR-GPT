import os
import json
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import numpy as np

import os
class LiDARGPTDataset(Dataset):
    def __init__(self, data_root, ann_file, n_points=10000, n_sweeps=10):
        """
        Khởi tạo Dataset cho MiniGPT-3D sử dụng dữ liệu nuScenes.

        Args:
            data_root (str): Đường dẫn gốc tới folder chứa dữ liệu nuScenes (raw data).
                             Ví dụ: '/data/nuscenes'
            ann_file (str): Đường dẫn tới file JSON metadata đã tạo ở Giai đoạn 1.
                            Ví dụ: './unified_nuscenes_infos.json'
            n_points (int): Số lượng điểm cố định cần sample cho mỗi scene (Uni3D yêu cầu).
                            Mặc định: 10,000.
            n_sweeps (int): Số lượng frame quá khứ cần tích lũy.
                            Mặc định: 10 (tương đương 0.5 giây với LiDAR 20Hz).
        """
        super().__init__()
        
        # 1. Lưu các tham số cấu hình
        self.data_root = data_root
        self.n_points = n_points
        self.n_sweeps = n_sweeps

        # 2. Load dữ liệu Metadata (File JSON)
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Không tìm thấy file metadata tại: {ann_file}")
        
        print(f"Đang tải metadata từ: {ann_file}...")
        with open(ann_file, 'r') as f:
            self.data_infos = json.load(f)
        
        print(f"Đã tải thành công {len(self.data_infos)} mẫu dữ liệu.")

        # 3. Định nghĩa Prompt Template chuẩn của MiniGPT-3D
        # Token <PC>... là tín hiệu để model biết vị trí chèn embedding 3D
        self.prompt_template = "<PC><PointCloudHere></PC> {}"

    def __len__(self):
        return len(self.data_infos)

    def load_pc(self, path):
        """ Đọc file binary .bin của nuScenes.
        Input: Đường dẫn file (tương đối).
        Output: Numpy array shape (N, 4) gồm [x, y, z, intensity].
        """
        full_path = os.path.join(self.data_root, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        
        # Đọc binary float32
        points = np.fromfile(full_path, dtype=np.float32)
        points = points.reshape(-1, 5)
        # Chỉ lấy x, y, z, intensity (bỏ ring_index ở cột 5)
        return points[:, :4]

    def get_matrix(self, pose):
        """Hàm helper chuyển translation/rotation thành ma trận 4x4"""
        trans = np.array(pose['translation'])
        rot = Quaternion(pose['rotation']).rotation_matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        matrix[:3, 3] = trans
        return matrix
    
    def accumulate_sweeps(self, index):
        info = self.data_infos[index]
        
        # 1. Load Frame Hiện tại (Reference)
        ref_points = self.load_pc(info['lidar_path'])
        # Thêm time_lag = 0
        ref_points = np.hstack([ref_points, np.zeros((ref_points.shape[0], 1), dtype=np.float32)])
        
        # --- CHUẨN BỊ MA TRẬN CHO REF FRAME ---
        # Ref Lidar -> Ref Ego
        ref_calib = info['calibrated_sensor']
        ref_lidar2ego = self.get_matrix(ref_calib)
        
        # Ref Ego -> Global
        ref_pose = info['ego_pose']
        ref_ego2global = self.get_matrix(ref_pose)
        
        # Tính ma trận đảo: Global -> Ref Lidar 
        # (Global -> Ego -> Lidar) = inv(Ref Ego -> Global) * inv(Ref Lidar -> Ref Ego)
        ref_global2ego = np.linalg.inv(ref_ego2global)
        ref_ego2lidar = np.linalg.inv(ref_lidar2ego)
        
        # Gom lại: Global -> Ref Lidar
        global2ref_lidar = np.dot(ref_ego2lidar, ref_global2ego)

        all_points_list = [ref_points]

        # 2. Loop Sweeps
        if 'sweeps' in info:
            for i, sweep in enumerate(info['sweeps']):
                if i >= self.n_sweeps: break
                
                sweep_points = self.load_pc(sweep['lidar_path'])
                if sweep_points.shape[0] == 0: continue
                
                # --- CHUYỂN HỆ TỌA ĐỘ SWEEP ---
                # A. Sweep Lidar -> Sweep Ego
                sweep_calib = sweep['calibrated_sensor']
                sweep_lidar2ego = self.get_matrix(sweep_calib)
                
                # B. Sweep Ego -> Global
                sweep_pose = sweep['ego_pose']
                sweep_ego2global = self.get_matrix(sweep_pose)
                
                # C. Tổng hợp: Sweep Lidar -> Global
                sweep_lidar2global = np.dot(sweep_ego2global, sweep_lidar2ego)
                
                # D. Tổng hợp cuối cùng: Sweep Lidar -> Ref Lidar
                # P_ref = (Global -> Ref Lidar) * (Sweep Lidar -> Global) * P_sweep
                transform_matrix = np.dot(global2ref_lidar, sweep_lidar2global)
                
                # Thực hiện biến đổi nhân ma trận
                # Points: (N, 3) -> Thêm cột 1 -> (N, 4) để nhân với ma trận 4x4
                points_xyz = sweep_points[:, :3]
                num_points = points_xyz.shape[0]
                points_homo = np.hstack([points_xyz, np.ones((num_points, 1))])
                
                # Nhân: (4, 4) . (4, N) -> (4, N) -> Transpose lại thành (N, 4)
                points_transformed = np.dot(transform_matrix, points_homo.T).T
                
                # Cập nhật lại XYZ mới
                sweep_points[:, :3] = points_transformed[:, :3]
                
                # Thêm Time lag
                time_lag = np.ones((sweep_points.shape[0], 1), dtype=np.float32) * (i + 1)
                sweep_points = np.hstack([sweep_points, time_lag])
                
                all_points_list.append(sweep_points)
                
        return np.concatenate(all_points_list, axis=0)
    def filter_range(self, points):
        """
        Lọc bỏ các điểm nằm ngoài phạm vi quan tâm.
        Giúp loại bỏ nhiễu ở xa và giảm kích thước dữ liệu.
        """
        # Cấu hình range (theo LiDAR-LLM/PointPillars): [-54m, 54m] cho X, Y
        # Z thường lấy từ [-5m, 3m]
        x_min, x_max = -54.0, 54.0
        y_min, y_max = -54.0, 54.0
        z_min, z_max = -5.0, 3.0
        
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
               (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
               
        return points[mask]
    
    def uni3d_process(self, points):
        """
        Xử lý hình học theo chuẩn Uni3D
        
        Input: Numpy array (N, C) từ bước filter.
        Output: Numpy array (10000, 6) gồm [x, y, z, r, g, b].
        """
        # --- A. Sampling (Bắt buộc vì nuScenes raw có số điểm thay đổi) ---
        # Uni3D yêu cầu cố định 10,000 điểm.
        num_points = points.shape[0]
        
        if num_points == 0:
            # Fallback an toàn: trả về array rỗng đúng shape
            return np.zeros((self.n_points, 6), dtype=np.float32)

        if num_points >= self.n_points:
            # Dư điểm -> Chọn ngẫu nhiên không hoàn lại (nhanh hơn FPS)
            choices = np.random.choice(num_points, self.n_points, replace=False)
        else:
            # Thiếu điểm -> Chọn ngẫu nhiên có hoàn lại (Padding)
            choices = np.random.choice(num_points, self.n_points, replace=True)
        
        # Lấy xyz đã sample
        xyz = points[choices, :3] # Chỉ lấy 3 cột đầu (XYZ), bỏ intensity/time cũ

        # --- B. Normalization (Unit Sphere) ---
        # Theo code Uni3D: pc = (pc - mean) / max_norm
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        # Tính khoảng cách Euclidean xa nhất từ tâm
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        
        # Guard: tránh chia cho 0
        if m > 0:
            xyz = xyz / m

        # --- C. Input Channels (Quan trọng: Thêm RGB giả) ---
        # Uni3D encoder mong đợi đầu vào 6 kênh (XYZ + RGB).
        # Với dữ liệu không màu (như LiDAR), Uni3D fill giá trị 0.4.
        rgb = np.ones_like(xyz) * 0.4
        
        # Gộp lại thành tensor (10000, 6)
        points_6c = np.concatenate((xyz, rgb), axis=1)
            
        return points_6c 
    

    def __getitem__(self, index):
        """
        Đóng gói dữ liệu đầu ra theo chuẩn MiniGPT-3D.
        """
        # 1. Lấy thông tin metadata
        item = self.data_infos[index]
        
        # --- XỬ LÝ POINT CLOUD ---
        # A. Tích lũy 10 frame (Logic nuScenes)
        # Hàm này trả về numpy array (N, 5)
        raw_points = self.accumulate_sweeps(index)
        
        # B. Lọc nhiễu không gian
        # Hàm này trả về numpy array (M, 5)
        filtered_points = self.filter_range(raw_points)
        
        # C. Chuẩn hóa theo Uni3D (Quan trọng nhất)
        # Hàm này trả về numpy array (10000, 6) gồm [x, y, z, 0.4, 0.4, 0.4]
        processed_points = self.uni3d_process(filtered_points)
        
        # D. Chuyển sang Tensor
        pc_tensor = torch.from_numpy(processed_points.astype(np.float32))

        # --- XỬ LÝ TEXT ---
        # Lấy instruction và answer từ file JSON đã chuẩn bị
        raw_instruction = item.get('instruction', "")
        raw_answer = item.get('answer', "")
        
        # E. Format Instruction 
        instruction_input = self.prompt_template.format(raw_instruction)

        # --- RETURN DICTIONARY ---
        # Output trả về đúng các key mà MiniGPT-3D model yêu cầu trong forward()
        return {
            "pc": pc_tensor,                 # Tensor [10000, 6]
            "instruction_input": instruction_input, # "<PC><PointCloudHere></PC> Describe..."
            "answer": raw_answer,            # "There is a car..."
            "PC_id": item.get('sample_token', str(index)) # Token để tracking/debug
        }

if __name__ == "__main__":
    import sys
    
    # --- CẤU HÌNH TEST ---
    # Thay đường dẫn này bằng đường dẫn thật để test
    # Nếu chưa có data thật, code sẽ báo lỗi FileNotFound
    DATA_ROOT = "./data/nuscenes"
    ANN_FILE = "./unified_nuscenes_infos.json"
    
    print(f"🚀 Bắt đầu Sanity Check...")
    print(f"📂 Data Root: {DATA_ROOT}")
    print(f"📄 Ann File: {ANN_FILE}")

    # 1. Thử khởi tạo Dataset
    try:
        dataset = LiDARGPTDataset(
            data_root=DATA_ROOT,
            ann_file=ANN_FILE,
            n_points=10000,
            n_sweeps=10
        )
        print(f"✅ Khởi tạo thành công! Tổng số mẫu: {len(dataset)}")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        sys.exit(1)

    # 2. Lấy thử mẫu đầu tiên
    try:
        sample = dataset[0]
        pc = sample['pc']
        instr = sample['instruction_input']
        ans = sample['answer']
        token = sample['PC_id']

        print("\n--- 🔍 Kiểm tra Mẫu số 0 ---")
        print(f"🆔 Token ID: {token}")
        
        # 3. Check Point Cloud Shape
        # Kỳ vọng: [10000, 6] (XYZ + RGB giả)
        print(f"📦 PC Shape: {pc.shape}")
        if pc.shape == (10000, 6):
            print("   ✅ Shape chuẩn (10k điểm, 6 kênh).")
        else:
            print(f"   ⚠️ Cảnh báo: Shape lạ, kỳ vọng (10000, 6).")

        # 4. Check Normalization
        # Kỳ vọng: Giá trị nằm trong khoảng [-1, 1] (hoặc lân cận)
        xyz = pc[:, :3]
        max_val = torch.max(xyz).item()
        min_val = torch.min(xyz).item()
        print(f"📊 PC Range (XYZ): Min={min_val:.4f}, Max={max_val:.4f}")
        
        if -1.1 <= min_val and max_val <= 1.1:
            print("   ✅ Normalization có vẻ đúng (nằm trong Unit Sphere).")
        else:
            print("   ⚠️ Cảnh báo: Giá trị vượt quá [-1, 1], kiểm tra lại logic normalize.")

        # 5. Check Instruction Format
        print(f"📝 Instruction: \"{instr}\"")
        if "<PC><PointCloudHere></PC>" in instr:
            print("   ✅ Format chuẩn MiniGPT-3D.")
        else:
            print("   ❌ Lỗi: Thiếu thẻ <PC>... trong instruction!")

        print(f"🗣️ Answer: \"{ans}\"")

        print("\n🎉 CHÚC MỪNG! Dataset Class hoạt động ổn định.")

    except Exception as e:
        print(f"\n❌ Lỗi khi lấy mẫu: {e}")
        import traceback
        traceback.print_exc()

    try:
        import open3d as o3d
        print("\n🎨 Đang hiển thị Point Cloud (Cửa sổ 3D sẽ hiện ra)...")
        
        # Lấy xyz từ tensor
        xyz = pc[:, :3].numpy()
        
        # Tạo object Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # Thêm trục tọa độ để dễ nhìn (Red=X, Green=Y, Blue=Z)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        
        # Hiển thị
        o3d.visualization.draw_geometries([pcd, axes], window_name="Check Normalize")
        print("✅ Visualize xong. Nếu thấy đám mây điểm hình cầu nằm gọn quanh gốc tọa độ là đúng!")
        
    except ImportError:
        print("⚠️ Chưa cài open3d nên không visualize được. (pip install open3d)")
