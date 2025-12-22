"""
전처리 파이프라인 모듈.

AI 음악 탐지 모델을 위한 데이터 전처리 및 변환 함수들을 제공합니다.
Sunday 논문(2025) 재현을 위한 ImageNet 정규화와 ResNet18 입력 변환을 포함합니다.
"""

import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import json
import random
from sklearn.model_selection import train_test_split

def normalize_for_imagenet(tensor: Union[np.ndarray, Tensor]) -> Tensor:
    """
    ImageNet 정규화를 사용합니다.

    Sunday 논문에서 사용한 ImageNet1k v1 pretrained weights의 정규화 값을 적용합니다.

    Args:
        tensor: 정규화할 텐서 (C, H, W) 또는 (B, C, H, W) 형태
                값의 범위는 [0, 1]이어야 함

    Returns:
        정규화된 torch.Tensor

    Note:
        ImageNet 정규화 값:
        - mean = [0.485, 0.456, 0.406]
        - std = [0.229, 0.224, 0.225]

    Examples:
        >>> spec = np.random.rand(3, 224, 224)
        >>> normalized = normalize_for_imagenet(spec)
        >>> print(normalized.shape)
        torch.Size([3, 224, 224])
    """
    # NumPy 배열을 Tensor로 변환
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    else:
        tensor = tensor.float()

    # ImageNet 정규화 값
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # 배치 차원이 있는 경우
    if tensor.dim() == 4: # (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.dim() == 3: # (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    else:
        raise ValueError(f"텐서는 3차원 (C,H,W) 또는 4차원 (B,C,H,W)이어야 합니다. 현재: {tensor.shape}")
    
    # 정규화 적용
    normalized = (tensor - mean) / std

    return normalized


def prepare_mel_for_resnet18(mel_spec: np.ndarray, target_size: Tuple[int, int] =(224, 224)) -> np.ndarray:
    """
    Mel spectrogram을 ResNet18 입력 형식으로 변환합니다.

    1채널 Mel spectrogram을 ResNet18이 요구하는 3채널 RGB 형식으로 변환합니다.
    논문에 명시되지 않은 부분은 일반적인 관행을 따릅니다.

    Args:
        mel_spec: Mel spectrogram (H, W)형태
        target_size: 목표 크기 (height, width), 기본값 (224, 224)

    Returns:
        3채널 RGB 형식 배열 (3, H, W)

    Note:
        채널 변환 방식 (실험 필요):
        1. 단순 복제: 1채널을 3번 복사
        2. 다른 표현 결합: Mel + Delta + Delta-Delta
        현재는 방식 1(단순 복제) 사용
    """
    # 입력 검증
    if mel_spec.ndim != 2:
        raise ValueError(f"Mel spectrogram은 2차원이어야 합니다. 현재: {mel_spec.shape}")
    
    # 크기 조정 (필요한 경우)
    if mel_spec.shape != target_size:
        from scipy import ndimage
        zoom_factors = (target_size[0] / mel_spec.shape[0],
                    target_size[1] / mel_spec.shape[1])
        mel_spec = ndimage.zoom(mel_spec, zoom_factors, order=1)

    # Min-Max 정규화 (0-1 범위로)
    mel_min = mel_spec.min()
    mel_max = mel_spec.max()
    if mel_max > mel_min:
        mel_spec = (mel_spec - mel_min) / (mel_max - mel_min)
    else:
        mel_spec = np.zeros_like(mel_spec)

    # 1채널 -> 3채널 변환 (단순 복제)
    # TODO: Delta, Delta-Delta 특징을 사용한 3채널 변환 실험 필요
    rgb_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)

    return rgb_spec

def split_fakemusiccaps(data_dir: Union[str, Path], metadata_file: Optional[str] = None, random_state: int = 42) -> Dict[str, List[Path]]:
    """
    FakeMusicCaps 데이터셋을 Sunday 논문 기준으로 분할합니다.

    논문 기준 분할:
    - Train: 8,599 샘플
    - Val: 1,075 샘플
    - Test: 1,074 샘플
    - 총: 10,748 샘플
    """
    data_dir = Path(data_dir)

    # 메타데이터 파일이 제공된 경우
    if metadata_file:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    # 메타데이터에서 split 정보가 있는 경우
    if 'splits' in metadata:
        return {
            'train': [Path(p) for p in metadata['splits']['train']],
            'val': [Path(p) for p in metadata['splits']['val']],
            'test': [Path(p) for p in metadata['splits']['test']]
        }

    # 오디오 파일 수집 (MP3, WAV, FLAC)
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(list(data_dir.rglob(ext)))

    # 파일 개수 검증
    total_files = len(audio_files)
    expected_total = 10748

    if total_files != expected_total:
        print(f"경고: 파일 개수({total_files})가 논문 기준({expected_total})과 다릅니다.")
        print(f"실제 비율로 분할합니다.: Train 80%, Val 10%, Test 10%")

    # 정렬하여 일관성 유지
    audio_files.sort()

    # 랜덤 시드 설정
    random.seed(random_state)
    np.random.seed(random_state)

    # sklearn으로 분할
    train_files, test_val_files = train_test_split(audio_files, test_size=0.2, random_state=random_state) # Val + Test
    val_files, test_files = train_test_split(test_val_files, test_size = 0.5, raondom_state=random_state) # Val과 Test를 반반

    # 정확한 개수로 조정 (가능한 경우)
    if total_files == expected_total:
        train_files = train_files[:8599]
        val_files = val_files[:1075]
        test_files = test_files[:1074]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # 분할 결과 출력
    print(f"데이터셋 분할 완료:")
    print(f"    Train: {len(splits['train'])} 샘플")
    print(f"    Val: {len(splits['val'])} 샘플")
    print(f"    Test: {len(splits['test'])} 샘플")
    print(f"    Total: {sum(len(s) for s in splits.values())} 샘플")

    return splits


def convert_to_db(spectrogram: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """
    Spectrogram을 dB 스케일로 변환합니다.

    Args:
        spectrogram: 선형 스케일 spectrogram
        ref: 참조값 (기본값: 1.0)
        amin: 최소값 (log 계산 시 0 방지)

    Returns:
        dB 스케일 spectrogram
    """
    # Power to dB
    db_spec = 10 * np.log10(np.maximum(amin, spectrogram))
    db_spec = np.maximum(db_spec, db_spec.max() - 80.0) # 80dB 다이나믹 레인지

    return db_spec


def prepare_batch(batch_specs: List[np.ndarray], use_imagenet_norm: bool = True) -> torch.Tensor:
    """
    배치 단위로 spectrogram을 처리합니다.

    Args:
        batch_specs: Mel spectrogram 리스트
        use_imagenet_norm: ImageNet 정규화 적용 여부

    Returns:
        배치 텐서 (B, 3, 224, 224)
    """
    processed = []

    for spec in batch_specs:
        # ResNet18 형식으로 변환
        rgb_spec = prepare_mel_for_resnet18(spec)
        processed.append(rgb_spec)

    # 배치로 결합
    batch_tensor = torch.from_numpy(np.stack(processed)).float()

    # ImageNet 정규화
    if use_imagenet_norm:
        batch_tensor = normalize_for_imagenet(batch_tensor)

    return batch_tensor

# 테스트용 메인 함수
if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("Preprocessing Module Test")
    print("=" * 50)

    # 1. ImageNet 정규화 테스트
    print("\n1. ImageNet Normalization Test:")
    dummy_tensor = np.random.rand(3, 224, 224)
    normalized = normalize_for_imagenet(dummy_tensor)
    print(f"    Imput shape: {dummy_tensor.shape}")
    print(f"    Output shape: {normalized.shape}")
    print(f"    Output type: {type(normalized)}")

    # 2. Mel spectrogram 변환 테스트
    print("\n2. Mel to ResNet18 Format Test:")
    dummy_mel = np.random.randn(128, 431) # 일반적인 Mel spectrogram 크기
    rgb_mel = prepare_mel_for_resnet18(dummy_mel)
    print(f"    Input shape: {dummy_mel.shape}")
    print(f"    Output shape: {rgb_mel.shape}")
    print(f"    Output range: [{rgb_mel.min():.3f}, {rgb_mel.max():.3f}]")

    # 3. 배치 처리 테스트
    print("\n3. Batch Processing Test:")
    batch_mels = [np.random.randn(128, 431) for _ in range(4)]
    batch_tensor = prepare_batch(batch_mels)
    print(f"    Batch size: {len(batch_mels)}")
    print(f"    Output shape: {batch_tensor.shape}")
    print(f"    Output dtype: {batch_tensor.dtype}")

    # 4. 데이터 분할 테스트 (실제 경로가 필요하므로 스킵)
    print("\n4. Dataset Split Test:")
    print(" (실제 데이터셋 경로가 필요하므로 스킵)")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
