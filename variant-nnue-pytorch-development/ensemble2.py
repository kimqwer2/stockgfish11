import torch
import copy
import warnings

# PyTorch 경고 무시 (weights_only 관련)
warnings.filterwarnings("ignore", category=FutureWarning)

def merge_models(model_A_path, model_B_path, output_path, alpha=0.5):
    """
    두 PyTorch 모델 객체(.pt)의 가중치를 alpha 비율로 병합합니다.
    """
    print(f"Loading Model A: {model_A_path}...")
    # 모델 전체 객체를 불러옵니다
    model_A = torch.load(model_A_path, map_location='cpu')
    
    print(f"Loading Model B: {model_B_path}...")
    model_B = torch.load(model_B_path, map_location='cpu')

    # 모델 객체에서 state_dict(가중치 딕셔너리)를 추출합니다
    state_dict_A = model_A.state_dict()
    state_dict_B = model_B.state_dict()

    # 새로운 C 모델을 담을 빈 가중치 딕셔너리 생성
    state_dict_C = copy.deepcopy(state_dict_A)

    print(f"Merging weights (Ratio -> A: {alpha:.1f} / B: {1.0 - alpha:.1f})...")
    # 두 모델의 모든 레이어(Layer)와 파라미터를 순회하며 평균값을 계산
    for key in state_dict_A.keys():
        if key in state_dict_B:
            # 공식: C = A * alpha + B * (1 - alpha)
            state_dict_C[key] = (state_dict_A[key] * alpha) + (state_dict_B[key] * (1.0 - alpha))
        else:
            print(f"Warning: Key {key} not found in Model B. Keeping Model A's weight.")

    # Model A 객체에 섞인 가중치(C)를 덮어씌웁니다
    model_A.load_state_dict(state_dict_C)

    # 파일로 저장 (모델 객체 전체 저장)
    print(f"Saving merged Model C to {output_path}...")
    torch.save(model_A, output_path)
    print("Done! You can now export this to .nnue format.")

if __name__ == "__main__":
    # 파일명은 에러 로그에 나온 그대로 설정했습니다.
    MODEL_A = "27.pt" 
    MODEL_B = "29.pt" 
    MODEL_C_OUTPUT = "3.pt"

    # 실행 (5:5 비율로 병합)
    merge_models(MODEL_A, MODEL_B, MODEL_C_OUTPUT, alpha=0.05)