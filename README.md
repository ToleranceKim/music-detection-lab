# Music Detection Lab

**AI 생성 음악 탐지 연구** - Sunday & Afchar 논문 재현 프로젝트

## 개요

이 레포지토리는 AI가 생성한 음악을 탐지하는 두 가지 주요 연구를 재현하고 비교합니다:

### 재현 논문

1. **Detecting Musical Deepfakes** (Nicholas Sunday, 2025)
  - CNN 기반 Mel 스펙트로그램 탐지
  - FakeMusicCaps 데이터셋 활용
  - 오디오 변조에 대한 강건성 연구

2. **AI-Generated Music Detection** (Darius Afchar/Deezer, ICASSP 2025)
  - 오토인코더 기반 탐지 접근법
  - 일반화 문제 해결
  - 스펙트로그램 기반 방법과의 비교


## 프로젝트 목표

- 두 논문의 최소 구현체(MVP) 재현
- 성능 및 강건성 비교 분석
- 통합 데모 애플리케이션 개발
- 한국어 문서화 및 개발 로그 작성

## 프로젝트 구조

```
music-detection-lab/
├── docs/
│   ├── development_plan.md  # 개발 계획서
│   ├── dev_log.md           # 개발 진행 로그
│   └── archive/             # 이전 문서 보관
│       └── research_plan.md # 전체 연구 계획 (참고용)
├── src/
│   ├── sunday/              # Sunday 논문 구현
│   │   ├── model.py        # ResNet18 기반 CNN
│   │   ├── dataset.py      # FakeMusicCaps 데이터 로더
│   │   └── train.py        # 학습 파이프라인
│   ├── afchar/             # Afchar 논문 구현
│   │   ├── model.py        # 오토인코더 아키텍처
│   │   ├── dataset.py      # 데이터 로더
│   │   └── train.py        # 학습 파이프라인
│   └── common/             # 공통 유틸리티
│       ├── audio_io.py     # 오디오 로딩/전처리
│       ├── features.py     # 특징 추출 (Mel spectrogram 등)
│       └── augment.py      # 데이터 증강 (pitch, tempo 변조)
├── experiments/            # 실험 스크립트
├── notebooks/              # Jupyter 노트북 (탐색적 분석)
├── demos/                  # Streamlit 데모
├── data/                   # 데이터셋 (gitignore)
├── models/                 # 학습된 모델 체크포인트 (gitignore)
├── requirements.txt        # pip 의존성
├── pyproject.toml         # 프로젝트 설정 (uv 사용)
└── uv.lock               # uv 락 파일
```

## 시작하기

### 필수 요구사항

- Python 3.12+
- PyTorch 2.2+
- CUDA 11.8+ (GPU 사용 시, 선택사항)
- 최소 8GB RAM (CPU 실행 시)
- 최소 10GB 디스크 공간 (데이터셋 포함)

### 설치
```bash
# 저장소 클론
git clone https://github.com/TeleranceKim/music-detection-lab.git
cd music-detection-lab

# Python 3.12 설정 (pyenv 사용 시)
pyenv local 3.12

# uv 설치 (아직 없는 경우)
pip install uv

# 가상환경 생성 및 패키지 설치
uv venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
uv pip sync requirements.txt

# 설치 확인
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import librosa; print(f'librosa {librosa.__version__}')"

pip install -r requirements.txt
```

## 개발 현황

개발 로드맵과 진행 상황은 다음 문서를 참고하세요:
- [개발 계획서](docs/development_plan.md) - 기술적 구현 세부사항
- [개발 로그](docs/dev_log.md) - 일별 진행 상황 및 이슈 트래킹

## 참고 문헌

### 논문
1. Nicholas Sunday, "Detecting Musical Deepfakes" (2025)
   - [arXiv:2505.09633](https://arxiv.org/abs/2505.09633)
   - [GitHub](https://github.com/nicksunday/deepfake-music-detector)

2. Darius Afchar et al., "AI-Generated Music Detection" (ICASSP 2025)
   - [arXiv:2405.04181](https://arxiv.org/abs/2405.04181)
   - [GitHub](https://github.com/deezer/deepfake-detector)

### 데이터셋
- **FakeMusicCaps**: AI 생성 음악 데이터셋
  - [논문](https://arxiv.org/abs/2409.10684)
  - [GitHub](https://github.com/polimi-ispl/FakeMusicCaps)
- **MusicCaps**: Google AudioSet 기반 원본 음악

## 라이센스

이 프로젝트는 MIT 라이센스하에 배포되지만, **비상업적 용도로만** 사용 가능합니다.

### 제한사항
- Deezer Research의 코드가 CC-BY-NC-4.0 라이센스로 포함되어 있어, 전체 프로젝트가 비상업적 용도로 제한됩니다

### 원저작물
- Nicholas Sunday의 코드: MIT License
- Deezer Research의 코드: CC-BY-NC-4.0

자세한 내용은 [LICENSE](LICENSE) 및 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) 파일을 참고하세요.
