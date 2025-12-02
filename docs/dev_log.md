# Development Log - Music Detection Lab

## 개요
AI 생성 음악 탐지 연구 프로젝트의 개발 진행 상황을 기록합니다.

---

## 2025년 12월

### 2025-12-01 (일)
**프로젝트 구조 재정의 및 개발 계획 완성**


#### 이전 작업
- [x] 프로젝트 범위를 music-detection-lab으로 재구성
- [x] README.md 한국어로 재작성
- [x] 라이센스 파일 추가 (MIT with non-commercial)
- [x] 패키지 관리를 uv로 통일

#### 완료된 작업
- [x] development_plan.md 완성
- [x] 프로젝트 디렉토리 구조 생성
    - src/common, src/sunday, src/afchar
    - experiments, demos 폴더
    - data, models 폴더 (gitignore 추가)
- [x] research_plan.md를 archive로 이동
    - 향후 참고용으로 보관

#### 이슈/결정사항
- Deezer 코드의 CC-BY-NC-4.0 라이센스로 인해 비상업적 용도 제한
- FakeMusicCaps 데이터셋 다운로드 확인 필요
- GPU 환경 없이 CPU로 초기 개발 진행 예정

---

### 2025-12-02 (월)
**Phase 2: 공통 유틸리티 모듈 구현 시작 & 프로젝트 재구성**

#### 완료된 작업
- [x] 프로젝트명 변경 완료
    - music-detection-lab -> ai-music-detection-lab
    - 디렉토리명, GitHub 저장소명, pyproject.toml 모두 통일
- [x] 환경 재구성
    - Python 3.12.12 + uv 환경으로 완전 재설정
    - 149개 패키지 의존성 설치 완료
- [x] audio_io.py 구현 완료
    - load_audio(): 다양한 포멧 지원, 자동 리샘플링
    - save_audio(): WAV 포맷 저장, 샘플레이트 지정
    - normalize_audio(): Peak/RMS 정규화 구현
    - split_audio(): 세그먼트 분할 기능
- [x] test_audio_io.py 테스트 작성 및 통과
    - 모든 함수에 대한 단위 테스트 구현
    - 정규화, 분할, 저장/로드 기능 검증 완료

#### 다음 작업
- [] Phase 2 시작: 공통 유틸리티 모듈 구현
    - [] features.py 작성
    - [] augment.py 작성
    - [] metrics.py 작성

#### 이슈/결정사항
- 가상환경 이름과 프로젝트명 통일
- 환경 설정: macOS 15.5 pyenv 빌드 실패 -> Anaconda Python 3.12.2 활용
- 오디오 라이브러리 : librosa

---

### 2025-12-03 (화) [예정]
**Phase 2 계속: 특징 추출 모듈 구현**

#### 목표
- [] features.py 구현
    - MFCC (Mel-frequency cepstral coefficients)
    - Spectral centroid, rolloff, contrast
    - Zero-crossing rate
    - Chroma features
- [] 단위 테스트 작성
- [] 문서화 및 사용 예제 추가

---

## 주간 요약

### Week 1 (2025-11-25 ~ 2025-12-01)
- 프로젝트 초기 설정 o
- 개발 계획 수립 o
- 프로젝트 구조 생성 o

---

## 마일스톤 트래킹

| 마일스톤 | 목표일 | 상태 | 완료일 |
|--------|------|-----|-------|
| 프로젝트 설정 | 2025-11-28 | o | - |
| 개발 계획 수립 | 2025-12-01 | o | - |
| 공통 모듈 완성 | - | - | - |
| Sunday 재현 | - | - | - |
| Afchar 재현 | - | - | - |
| 통합 및 비교 | - | - | - |
| 데모 완성 | - | - | - |

---

## 기술 메모

### FakeMusicCaps 데이터셋
- 총 27,605 트랙 (약 77시간)
- 10초 클립, 16kHz, mono
- 5개 TTM 모델 출력 포함
- [다운로드 링크 확인 필요]

### 주요 설정값
- Sunday 논문 설정: `src/sunday/config.py` 참조
- Afchar 논문 설정: `src/afchar/config.py` 참조

## 참고 링크

### 논문
- [Sunday Paper](https://arxiv.org/abs/2505.09633)
- [Afchar Paper](https://arxiv.org/abs/2405.04181)

### GitHub
- [프로젝트 저장소](https://github.com/ToleranceKim/music-detection-lab)
- [Sunday 원본 코드](https://github.com/nicksunday/deepfake-music-detector)
- [Deezer 원본 코드](https://github.com/deezer/deepfake-detector)

### 데이터셋
- [FakeMusicCaps](https://github.com/polimi-ispl/FakeMusicCaps)
- [FMA Dataset](https://github.com/mdeff/fma)

---
