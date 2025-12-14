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

