import h5py
import os

def print_h5_structure(file_path):
    """
    .h5 파일을 열어 내부의 그룹(Group)과 데이터셋(Dataset) 구조를
    트리 형태로 출력합니다.
    """
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n📦 File: {os.path.basename(file_path)}")
            print("=" * 40)
            
            # visititems는 파일 내의 모든 객체를 방문하며 콜백 함수를 실행합니다.
            f.visititems(print_tree_info)
            
            print("=" * 40)
            print("출력 완료.")
            
    except Exception as e:
        print(f"파일을 여는 중 오류가 발생했습니다: {e}")

def print_tree_info(name, obj):
    """
    visititems 함수에 의해 호출되는 콜백 함수입니다.
    name: 객체의 경로 (예: group1/subgroup/data)
    obj: 실제 객체 (Group 또는 Dataset)
    """
    
    # 들여쓰기 수준 결정 (경로의 깊이에 따라 들여쓰기)
    depth = name.count('/')
    indent = "    " * depth
    
    # 경로에서 마지막 이름만 추출 (예: group1/data -> data)
    real_name = name.split('/')[-1]

    if isinstance(obj, h5py.Group):
        # 그룹일 경우 (폴더와 유사)
        print(f"{indent}📁 {real_name} (Group)")
        
    elif isinstance(obj, h5py.Dataset):
        # 데이터셋일 경우 (실제 데이터 파일과 유사)
        # 데이터의 차원(Shape)과 타입(dtype)을 함께 표시
        print(f"{indent}📄 {real_name} (Dataset) | Shape: {obj.shape}, Type: {obj.dtype}")

# ==========================================
# 사용 방법: 아래 경로를 실제 .h5 파일 경로로 수정하세요.
# ==========================================
if __name__ == "__main__":
    # 예: "model_weights.h5" 또는 "./data/my_data.h5"
    target_file = r"F:\coding자료\coding\digital_hearth_care\model_2\dataset_10sec.h5" 
    
    # 테스트를 위해 더미 파일이 없으면 생성 (실제 사용시에는 이 줄을 지우세요)
    # create_dummy_h5(target_file) 
    
    print_h5_structure(target_file)