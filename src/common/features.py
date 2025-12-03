"""
Audio Feature Extraction Module

AI 생성 음악 탐지를 위한 오디오 특징 추출 함수들
Sunday와 Afchar 논문에서 사용된 특징들 구현

주요 특징:
- MFCC: 음색 정보 (사람의 청각 시스템 모방)
- Spectral Features: 주파수 도메인 특성
- Temporal Features: 시간 도메인 특성
- Chroma: 음계 정보 (화성 구조)
"""

import numpy as np
import librosa
from typing import Optional, Tuple, Dict, List, Union
import warnings

# 1. MFCC

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    MFCC (Mel-frequency cepstral coefficients) 추출

    음성/음악의 가장 중요한 특징 중 하나로, 사람의 청각 시스템을 모방하여 음색(timbre) 정보를 효과적으로 표현합니다.
    AI 생성 음악은 종종 미묘하게 다른 MFCC 패턴을 보입니다.

    Args:
        audio: 오디오 신호 (1차원 numpy 배열)
        sr: 샘플레이트 (Hz)
        n_mfcc: MFCC 계수 개수 (default: 13)
        n_fft: FFT 윈도우 크기 (주파수 해상도 결정)
        hop_length: 프레임 간 이동 간격 (시간 해상도 결정)

    Returns:
        dict: MFCC 특징들
            - 'mfcc_mean': 각 계수의 시간축 평균 (shape: n_mfcc,)
            - 'mfcc_std': 각 계수의 시간축 표준편차 (shape: n_mfcc,)
            - 'mfcc_raw': 전체 MFCC 시퀀스 (shape: n_mfcc x time_frames)

    Example:
        >>> audio, sr = librosa.load('music.wav')
        >>> mfcc_features = extract_mfcc(audio, sr)
        >>> print(mfcc_features['mfcc_mean'].shape) # (13,)
    """

    # MFCC 계산
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'mfcc_raw': mfccs
    }


# 2. Spectral Features (centroid, rolloff, contrast)

def extract_spectral_centroid(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> Dict[str, float]:
    """
    Spectral Centroid (스펙트럼 중심) 추출

    주파수 스펙트럼의 "무게 중심"으로, 음색의 밝기를 나타냅니다.
    높은 값은 밝은 소리(고주파 많음), 낮은 값은 어두운 소리(저주파 많음)를 의미합니다.

    Args:
        audio: 오디오 신호
        sr: 샘플레이트
        n_fft: FFT 윈도우 크기
        hop_length: 프레임 간격

    Returns:
        dict: Spectral centroid 통계
            - 'spectral_centroid_mean': 평균값
            - 'spectral_centroid_std': 표준편차
    """
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0] # [0]으로 1차원 배열 추출

    return {
        'spectral_centroid_mean': np.mean(centroid),
        'spectral_centroid_std': np.std(centroid)
    }


def extract_spectral_rolloff(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, roll_percent: float = 0.85) -> Dict[str, float]:
    """
    Spectral Rolloff (스펙트럼 롤오프) 추출

    전체 스펙트럼 에너지의 85%가 포함되는 주파수 지점을 찾습니다.
    AI 음악의 주파수 분포 특성을 파악하는데 유용합니다.

    Args:
        audio: 오디오 신호
        sr: 샘플레이트
        n_fft: FFT 윈도우 크기
        hop_length: 프레임 간격
        roll_percent: 에너지 누적 비율 (기본 0.85)

    Returns:
        dict: Rolloff 통계
    """
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent)[0]

    return {
        'spectral_rolloff_mean': np.mean(rolloff),
        'spectral_rolloff_std': np.std(rolloff)
    }


def extract_spectral_contrast(audio: np.ndarray, sr:int, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 6) -> Dict[str, np.ndarray]:
    """
    Spectral Contrast (스펙트럼 대비) 추출

    각 주파수 대역에서 피크와 밸리의 차이를 측정합니다.

    Args:
        audio: 오디오 신호
        sr: 샘플레이트
        n_fft: FFT 윈도우 크기
        hop_length: 프레임 간격
        n_bands: 주파수 대역 수 (기본 6개)

    Returns:
        dict: 각 대역별 contrast 평균과 표준편차
    """
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)

    return {
        'spectral_contrast_mean': np.mean(contrast, axis=1),
        'spectral_contrast_std': np.std(contrast, axis=1)
    }


# 3. Temporal Features (zcr, rms)

def extract_zero_crossing_rate(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> Dict[str, float]:
    """
    Zero Crossing Rate (영교차율) 추출

    신호가 0을 교차하는 빈도를 측정합니다.
    타악기는 높은 ZCR, 현악기는 낮은 ZCR을 보입니다.

    Args:
        audio: 오디오 신호
        frame_length: 프레임 크기
        hop_length: 프레임 이동 간격

    Returns:
        dict: ZCR 통계
    """
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]

    return {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr)
    }

def extract_rms_energy(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> Dict[str, float]:
    """
    RMS Energy (Root Mean Square 에너지) 추출

    오디오 신호의 에너지를 측정합니다.

    Args:
        audio: 오디오 신호
        frame_length: 프레임 크기
        hop_length: 프레임 이동 간격

    Returns:
        dict: RMS 에너지 통계
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    return {
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms)
    }


# 4. Chroma Features
def extract_chroma(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = 12) -> Dict[str, np.ndarray]:
    """
    Chroma Features (크로마 특징) 추출

    12개 반음계의 에너지 분포를 나타냅니다.
    화성 구조 분석에 중요합니다.

    Args:
        audio: 오디오 신호
        sr: 샘플레이트
        n_fft: FFT 윈도우 크기
        hop_length: 프레임 간격
        n_chroma: 크로마 빈 개수 (12개 반음)

    Returns:
        dict: 각 음계별 평균과 표준편차
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)

    return {
        'chroma_mean': np.mean(chroma, axis=1),
        'chroma_std': np.std(chroma, axis=1)
    }


# 5. 통합 함수 구현

def extract_all_features(audio: np.ndarray, sr: int, feature_types: Optional[List[str]] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    모든 특징 추출 통합 함수

    선택한 특징들을 한번에 추출합니다.

    Args:
        audio: 오디오 신호
        sr: 샘플레이트
        feature_types: 추출할 특징 리스트 (None이면 모두)

    Returns:
        dict: 모든 추출된 특징들
    """
    if feature_types is None:
        feature_types = [
            'mfcc', 'spectral_centroid', 'spectral_rolloff',
            'spectral_contrast', 'zcr', 'rms', 'chroma'
        ]

    features = {}

    # MFCC
    if 'mfcc' in feature_types:
        mfcc_features = extract_mfcc(audio, sr)
        features['mfcc_mean'] = mfcc_features['mfcc_mean']
        features['mfcc_std'] = mfcc_features['mfcc_std']

    # Spectral features
    if 'spectral_centroid' in feature_types:
        features.update(extract_spectral_centroid(audio, sr))

    if 'spectral_rolloff' in feature_types:
        features.update(extract_spectral_rolloff(audio, sr))

    if 'spectral_contrast' in feature_types:
        contrast = extract_spectral_contrast(audio, sr)
        features['spectral_contrast_mean'] = contrast['spectral_contrast_mean']
        features['spectral_contrast_std'] = contrast['spectral_contrast_std']

    # Temporal features
    if 'zcr' in feature_types:
        features.update(extract_zero_crossing_rate(audio))

    if 'rms' in feature_types:
        features.update(extract_rms_energy(audio))

    # Chroma
    if 'chroma' in feature_types:
        chroma = extract_chroma(audio, sr)
        features['chroma_mean'] = chroma['chroma_mean']
        features['chroma_std'] = chroma['chroma_std']

    return features
