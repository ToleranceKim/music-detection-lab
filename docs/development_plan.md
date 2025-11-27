# Development Plan - Music Detection Lab

AI 생성 음악 탐지 연구를 위한 기술 개발 계획서
- Nicholas Sunday "Detecting Musical Deepfakes" (2025)
- Darius Afchar "AI-Generated Music Detection" (2025)

---

## 프로젝트 개요

### 목표
1. 두 가지 주요 AI 음악 탐지 논문의 재현
2. 성능 비교 및 분석
3. 통합 데모 시스템 구축

### 기술 스택
- **Python**: 3.12
- **PyTorch**: 2.2.2
- **Audio**:librosa 0.11.0, torchaudio
- **Web**: Streamlit
- **Package Manager**: uv

---

## Phase 1: 환경 설정 (완료)

- python 3.12 환경 구성
- uv 패키지 매니저 설치
- pyproject.toml 작성
- 기본 의존성 설치 (PyTorch, librosa)
- Git 저장소 초기화

---

## Phase 2: 공통 유틸리티 모듈 (진행중)

### 2.1 오디오 I/O ('src/common/audio_io.py)

**목적**: 오디오 파일 로딩 및 전처리 표준화

**구현 함수**:
- `load_audio(file_path, sr=22050)` - 오디오 로딩 및 리샘플링
- `save_audio(audio, sr, file_path)` - 오디오 저장
- `split_audio(audio, sr, segment_duration=10)` - 세그먼트 분할
- `normarlize_audio(audio)` - 정규화

### 2.2 특징 추출 (`src/common/features.py`)

**목적**: Mel-spectrogram 및 기타 특징 추출

**구현 함수**:
- `extract_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512)`
- `extract_mfcc(audio, sr, n_mfcc=13)`
- `spectrogram_to_image(spec)` - CNN 입력용 변환

### 2.3 데이터 증강 (`src/common/augment.py`)

**목적**: 강건성 향상을 위한 오디오 변조

**구현 함수**:
- `pitch_shift(audio, sr, n_steps)` - 피치 변조 (-2 ~ +2 semitones)
- `time_stretch(audio, rate)` - 템포 변조 (0.9 ~ 1.1)
- `add_noise(audio, noise_factor=0.005)` - 노이즈 추가
- `random_augment(audio, sr)` - 랜덤 증강 조합

### 2.4 평가 메트릭 (`src/common/metrics.py`)

**목적**: 모델 성능 평가 표준화

**구현 함수**:
- `calculate_accuracy(predictions, labels)`
- `calculate_auc_roc(predictions, labels)`
- `calculate_eer(predictions, labels)` - Equal Error Rate
- `plot_confusion_matrix(predictions, labels)`

**예상 소요 시간**: 3-4일

---

## Phase 3: Sunday 논문 재현

### 3.1 데이터셋 준비

**FakeMusicCaps 데이터셋**
- 다운로드 및 구조 파악
- 메타데이터 파싱
- Train/Val/Test 분할 (60/20/20)

### 3.2 모델 구현 (`src/sunday/model.py`)

```python
class sundayDtector(nn.Module):
    def __init__(self):
        # ResNet18 backbone
        # Binary classification head
```

**아키텍처**:
- Backbone: ResNet18 (ImageNet pretrained)
- Input: Mel-spectrogram (128 x T)
- Output: Binary (Real/Fake)

### 3.3 데이터 로더 (`src/sunday/dataset.py`)

```python
class FakeMusicCapsDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        
```