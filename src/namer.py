import os

def rename_csv_files(target_directory):
    """
    주어진 디렉토리 내의 모든 .csv 파일을 찾아
    seg_0000000.csv, seg_0000001.csv 형태로 이름을 변경합니다.
    """
    
    # 1. 디렉토리가 실제로 존재하는지 확인
    if not os.path.exists(target_directory):
        print(f"오류: '{target_directory}' 경로를 찾을 수 없습니다.")
        return

    # 2. 해당 디렉토리의 모든 파일 목록 가져오기
    files = os.listdir(target_directory)
    
    # 3. .csv 확장자를 가진 파일만 골라내고, 이름을 기준으로 정렬하기
    # (정렬을 해야 뒤죽박죽이 아닌 순서대로 번호가 매겨집니다)
    csv_files = sorted([f for f in files if f.endswith('.csv')])
    
    # CSV 파일이 없는 경우 처리
    if not csv_files:
        print("해당 디렉토리에 변경할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 발견했습니다. 변환을 시작합니다...")

    # 4. 파일 이름 변경 반복문 (Loop)
    for index, filename in enumerate(csv_files):
        # 현재 파일의 전체 경로 (예: ./data/old_name.csv)
        old_path = os.path.join(target_directory, filename)
        
        # 새로운 파일 이름 생성 (포맷팅: seg_ + 7자리 숫자 + .csv)
        # {index:07d} -> 숫자를 7자리로 맞추고 빈 곳은 0으로 채움
        new_filename = f"seg_{index:07d}.csv"
        new_path = os.path.join(target_directory, new_filename)
        
        # 이름 변경 실행
        try:
            os.rename(old_path, new_path)
            print(f"[변경 완료] {filename} -> {new_filename}")
        except FileExistsError:
            print(f"[오류] {new_filename} 파일이 이미 존재하여 변경하지 못했습니다.")
        except Exception as e:
            print(f"[오류] {filename} 변경 중 문제 발생: {e}")

    print("\n모든 작업이 완료되었습니다.")

# ==========================================
# 사용 방법: 아래 경로를 실제 디렉토리 경로로 수정하세요.
# ==========================================
if __name__ == "__main__":
    # 예: r"C:\Users\Name\Documents\MyData" 또는 "./data"
    directory_path = r"F:\coding자료\coding\digital_hearth_care\test_set\IHD_30s" 
    
    rename_csv_files(directory_path)