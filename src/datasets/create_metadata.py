import json
import os
import sys
from nuscenes.nuscenes import NuScenes
from datasets import load_dataset

def create_unified_metadata(
    nusc_root, 
    nusc_qa_path, 
    output_path="unified_nuscenes_infos.json"
):
    print("Loading nuScenes DB...")
    # Khởi tạo nuScenes devkit
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=True)
    
    unified_data = []

    # --- 1. Xử lý NuCaption (Captioning Task) ---
    print("Processing NuCaption...")
    try:
        ds_caption = load_dataset("Senqiao/LiDAR-LLM-Nu-Caption", split="train")
        for item in ds_caption:
            token = item['sample_token']
            
            try:
                lidar_info = get_lidar_info(nusc, token)
            except KeyError:
                continue 
                
            unified_data.append({
                "task": "caption",
                "sample_token": token,
                "lidar_path": lidar_info['filename'],
                "ego_pose": lidar_info['ego_pose'],
                "calibrated_sensor": lidar_info['calibrated_sensor'], 
                "sweeps": lidar_info['sweeps'],
                "instruction": item['question'],
                "answer": item['answer_lidar'] 
            })
    except Exception as e:
        print(f"Skipping NuCaption due to error or missing data: {e}")

    # --- 2. Xử lý NuScenes-QA (QA Task) ---
    print("Processing NuScenes-QA...")
    
    if os.path.exists(nusc_qa_path):
        with open(nusc_qa_path, 'r') as f:
            qa_raw_data = json.load(f) 
        
        qa_list = qa_raw_data['questions'] 
        print(f"Found {len(qa_list)} QA pairs.")

        for item in qa_list:
            token = item['sample_token']
            try:
                lidar_info = get_lidar_info(nusc, token)
            except KeyError:
                continue
            
            unified_data.append({
                "task": "qa",
                "id": f"qa_{token}_{len(unified_data)}",
                "sample_token": token,
                "lidar_path": lidar_info['filename'],
                "ego_pose": lidar_info['ego_pose'],
                "calibrated_sensor": lidar_info['calibrated_sensor'], 
                "sweeps": lidar_info['sweeps'],
                "instruction": item['question'], 
                "answer": item['answer'],
                "meta_hop": item.get('num_hop'),
                "meta_type": item.get('template_type')
            })
    else:
        print(f"Warning: QA path {nusc_qa_path} not found.")

    # --- 3. Lưu file tổng hợp ---
    print(f"Saving {len(unified_data)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(unified_data, f, indent=2)

def get_lidar_info(nusc, sample_token):
    """Hàm phụ trợ lấy đường dẫn LiDAR, Pose và Calibration"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    
    # 1. Lấy Ego Pose (Vị trí xe vs Global)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # 2. Lấy Calibrated Sensor (Vị trí Lidar vs Xe) 
    calib_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Lấy danh sách sweeps
    sweeps = []
    curr_sd_rec = lidar_data
    
    for _ in range(10):
        if curr_sd_rec['prev'] == '':
            break
        curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
        
        sweep_ego = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
        sweep_calib = nusc.get('calibrated_sensor', curr_sd_rec['calibrated_sensor_token'])
        
        sweeps.append({
            "lidar_path": curr_sd_rec['filename'],
            "ego_pose": sweep_ego,
            "calibrated_sensor": sweep_calib 
        })
        
    return {
        "filename": lidar_data['filename'],
        "ego_pose": ego_pose,
        "calibrated_sensor": calib_sensor, 
        "sweeps": sweeps
    }

if __name__ == "__main__":
    # Đảm bảo đường dẫn này đúng trên máy bạn
    create_unified_metadata(
        nusc_root="./data/nuscenes", 
        nusc_qa_path="./data/nuscenesQA/NuScenes_train_questions.json"
    )