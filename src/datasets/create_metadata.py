import json
import os
from nuscenes.nuscenes import NuScenes
from datasets import load_dataset
import sys
def create_unified_metadata(
    nusc_root, 
    nusc_qa_path, 
    output_path="unified_nuscenes_infos.json"
):
    print("Loading nuScenes DB...")
    # Khởi tạo nuScenes devkit để tra cứu đường dẫn file từ token
    # TODO: Edit the correct version
    nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=True)
    
    unified_data = []

    # --- 1. Xử lý NuCaption (Captioning Task) ---
    print("Processing NuCaption...")
    # Tải trực tiếp từ Hugging Face
    ds_caption = load_dataset("Senqiao/LiDAR-LLM-Nu-Caption", split="train")
    
    for item in ds_caption:
        token = item['sample_token']
        
        # Lấy thông tin LiDAR path và Sweeps từ nuScenes SDK
        try:
            lidar_info = get_lidar_info(nusc, token)
        except KeyError:
            continue # Bỏ qua nếu token không khớp với version nuScenes hiện tại
            
        unified_data.append({
            "task": "caption",
            "sample_token": token,
            "lidar_path": lidar_info['filename'],
            "ego_pose": lidar_info['ego_pose'],
            "sweeps": lidar_info['sweeps'],
            # Dùng answer_lidar để tránh ảo giác màu sắc
            "instruction": item['question'],
            "answer": item['answer_lidar'] 
        })

    # --- 2. Xử lý NuScenes-QA (QA Task) ---
    print("Processing NuScenes-QA...")
    
    # Load file json QA (Cấu trúc bạn vừa gửi)
    with open(nusc_qa_path, 'r') as f:
        qa_raw_data = json.load(f) 
    
    # LƯU Ý: Dữ liệu nằm trong key "questions"
    qa_list = qa_raw_data['questions'] 

    print(f"Found {len(qa_list)} QA pairs.")

    for item in qa_list:
        # Lấy sample_token
        token = item['sample_token']
        origin_stdout = sys.stdout
        try:
            # Tra cứu thông tin LiDAR từ token bằng nuScenes SDK
            lidar_info = get_lidar_info(nusc, token)
            with open("output.txt", "w", encoding="utf-8") as f:
                sys.stdout = f
                print(lidar_info)
                print()

        except KeyError:
            # Bỏ qua nếu token không tìm thấy trong version nuScenes đang dùng (mini/trainval)
            continue
        sys.stdout = origin_stdout
        unified_data.append({
            "task": "qa",
            "id": f"qa_{token}_{len(unified_data)}", # Tạo ID duy nhất nếu cần debug
            "sample_token": token,
            
            # Thông tin để load Point Cloud
            "lidar_path": lidar_info['filename'],
            "ego_pose": lidar_info['ego_pose'],
            "sweeps": lidar_info['sweeps'],
            
            # Thông tin Input/Output cho mô hình
            # QA instruction: template + câu hỏi
            "instruction": item['question'], 
            "answer": item['answer'],
            
            # Lưu thêm metadata (để sau này đánh giá model theo độ khó)
            "meta_hop": item.get('num_hop'),
            "meta_type": item.get('template_type')
        })

    # --- 3. Lưu file tổng hợp ---
    print(f"Saving {len(unified_data)} samples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(unified_data, f, indent=2)

def get_lidar_info(nusc, sample_token):
    """Hàm phụ trợ lấy đường dẫn LiDAR và 10 frame cũ (sweeps)"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # Lấy danh sách sweeps (ngược thời gian)
    sweeps = []
    curr_sd_rec = lidar_data
    
    # Lấy 10 frame cũ hơn
    for _ in range(10):
        if curr_sd_rec['prev'] == '':
            break
        curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])
        sweeps.append({
            "lidar_path": curr_sd_rec['filename'],
            "ego_pose": nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
        })
        
    return {
        "filename": lidar_data['filename'],
        "ego_pose": ego_pose,
        "sweeps": sweeps
    }

# --- Chạy script ---
if __name__ == "__main__":
    # Thay đường dẫn thực tế của bạn vào đây
    create_unified_metadata(
        nusc_root="./data/nuscenes",
        nusc_qa_path="./data/nuscenesQA/NuScenes_train_questions.json"
    )
    # load_caption()