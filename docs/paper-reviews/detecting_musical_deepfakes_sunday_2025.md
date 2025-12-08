# Detecting Musical Deepfakes

- title: "[논문 리뷰] Detecting Musical Deepfakes – FakeMusicCaps와 ResNet18 기반 AI 음악 딥페이크 탐지"
- date: 2025-12-05
- tags: [AI Music Detection, Deepfake, FakeMusicCaps, ResNet18]
---

# Detecting Musical Deepfakes: FakeMusicCaps와 ResNet18으로 살펴본 음악 딥페이크 탐지

## 리뷰를 시작하며

첫 리뷰로써 AI Music Detection 쪽에서 구현과 실험 구성이 비교적 직관적인 논문을 함께 탐구해 보려고 합니다. 복잡한 모델과 방어 기법으로 바로 들어가기보다는, 재현 가능한 예제를 통해 “생성 음악을 기계가 어떻게 구분하는지”를 먼저 체감해 보는 것이 목표입니다.

오늘 다룰 Nicholas Sunday의 「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋의 오디오를 Mel Spectrogram으로 변환한 뒤 ResNet18 이진 분류기를 학습하는 기본적인 탐지 파이프라인을 제안합니다. 전체 파이프라인이 GitHub에 공개되어 있어 코드를 내려받아 곧바로 학습과 평가를 재현해 볼 수 있고, 실험 설정을 바꾸어 보기도 쉽습니다.

Sunday는 Afchar 등 Deezer 연구와 SONICS, FakeMusicCaps 논문을 주요 관련 연구로 검토하고, Deezer 연구에서 사용된 조작 시나리오를 바탕으로 피치 쉬프트와 템포 스트레치 같은 단순 효과가 탐지 성능을 얼마나 흔들 수 있는지를 FakeMusicCaps와 ResNet18 구성에서 다시 시험합니다.

이 리뷰에서는 파이프라인을 실제로 실행해 본 경험을 바탕으로, 이 구성이 AI Music Detection을 이해하기 위한 직관적인 출발점으로서 어느 정도까지 유효한지, 또 어떤 한계를 드러내는지를 정리합니다.

> 핵심 요약
> Sunday의  「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋과 Mel Spectrogram + ResNet18 이진 분류기를 사용해 사람vs딥페이크 음악 탐지 문제를 실험적으로 분석한 연구입니다. 논문에 따르면, 10초 단위 오디오 클립 10,746개(사람 5,373 / 딥페이크 5,373)를 학습과 평가에 사용했을 때 모든 실험에서 F1, Accuracy, Recall, Precision이 80%를 상회하는 비교적 높은 성능을 달성합니다. 또한 Deezer 연구를 참조해 피치 쉬프트와 템포 스트레치 같은 단순 조작이 탐지 성능을 얼마나 떨어뜨리는지 측정하고 여러 조작 데이터셋을 연속적으로 학습하는 Continuous Learning 설정이 사람 음악 재현율을 높이는 대신 오탐률을 크게 증가시키는 트레이드오프를 보여 줍니다.

---

## 논문 정보 (Paper Information)

|항목|내용|
|---|---|
|제목|Detecting Musical Deepfakes|
|저자|Nicholas Sunday, Department of Computer Science, The University of Texas at Austin, USA|
|소속|Department of Computer Science, The University of Texas at Austin, USA|
|유형|arXiv preprint (cs.SD, cs.LG), UT Austin coursework 기반 연구|
|논문 링크|[arXiv:2505.09633](https://arxiv.org/abs/2505.09633)|
|코드|[GitHub Repository](https://github.com/nicksunday/deepfake-music-detector)|
|주요 데이터셋|FakeMusicCaps[1], MusicLM 관련 MusicCaps[2]|
|주요 관련 연구|SONICS[3], "Detecting music deepfakes is easy but actually hard"[4]|

---

# 1. 연구 배경

## 1.1 Text-to-Music와 음악 딥페이크의 등장

논문에 따르면 최근 Text to Music 플랫폼의 발전으로, 비교적 짧은 텍스트 프롬프트만으로 사람 연주와 구분하기 어려운 음악을 누구나 생성할 수 있는 환경이 조성되었습니다. 이 연구는 이러한 음악을 기존 이미지, 음성 딥페이크와 같은 범주의 생성 미디어로 보고, 저작권 침해, 거짓 저자 표기, 예술적 진정성 침식 같은 문제를 정책 관점과 기술 관점에서 동시에 제기합니다.

특히 Sunday는 Deezer, FakeMusicCaps, SONICS 등 선행 연구에서 공통으로 드러난 긴장 관계를 강조합니다. 한편으로는 창작 접근성을 높이는 긍정적 가능성이 존재하지만, 다른 한편으로는 플랫폼 차원에서 딥페이크를 탐지하고 관리하는 기술적 수단이 없으면 음악 생태계 전반의 신뢰가 훼손될 수 있다는 점을 반복해서 지적합니다.
---

# 1.2 FakeMusicCaps와 MusicCaps

FakeMusicCaps는 Politecnico di Milano 연구진이 제안한 공개 데이터셋으로, MusicCaps의 텍스트 캡션을 기반으로 여러 Text to Music 모델이 생성한 딥페이크 음악을 모아 구축되었습니다[1, 2]. Sunday 논문에 따르면 FakeMusicCaps는 다음과 같은 구조를 가집니다.

- 사람 연주 10초 오디오 클립 5,373개
- 다섯 개 TTM 플랫폼(MusicGen, audioldm2, musicldm, mustango, stable_audio_open)이 생성한 10초 딥페이크 트랙 5,521개
- Suno 플랫폼에서 생성된 더 긴 딥페이크 트랙 63개

Sunday는 이 가운데 딥페이크 5,373개를 무작위로 선택해 사랍 5,373개와 짝을 맞추고, 총 10,746개의 샘플을 사람 대 딥페이크 이진 분류 문제로 재구성합니다. 이때 개별 TTM 플랫폼 식별은 목표로 삼지 않고, 플랫폼 전체를 하나의 딥페이크 클래스로 통합합니다. 이 설계 선택 덕분에 모델 구조와 실험 설정이 단순해지는 대신, 플랫폼별 차이를 활용한 세분화 분석은 포기하는 구도가 됩니다

# 1.3 SONICS와 Deezer 연구 속 Sunday 논문의 위치

SONICS 프로젝트는 Synthetic Or Not, Identifying Couterfeit Songs라는 이름 그대로, 전체 곡 단위의 딥페이크를 대상으로 SpecTTTra라는 새로운 아키텍처와 풀길이 데이터셋을 제안합니다[3]. 이 연구는 과거의 짧은 클립 기반 모델과 데이터셋이 구조적 다양성과 가사, 편곡, 곡 구조 측면에서 한계가 있다는 점을 지적하며, 전체 곡 단위의 문맥을 모델링하는 방향을 강조합니다.

반면 Deezer 연구(Afchar et al. 2024)는 Sunday 논문에 따르면 살마 음악과 딥페이크 음악을 CNN으로 분류하는 작업이 기본 설정에서는 비교적 쉽지만, 피치 쉬프트와 템포 스트레치 같은 단순 조작을 가하면 성능이 급격히 떨어진다는 사실을 보여줍니다[4]. 즉, 탐지 자체보다 조작에 대한 견고함이 문제의 핵심이라는 관점을 제공합니다.

Sunday의 Detecting Musical Deepfakes는 이 두 축 사이에서

- FakeMusicCaps와 ResNet18을 사용해 짧은 10초 클립 수준에서 이진 분류 기준선을 세우고
- Deezer 연구에서 제시한 피치 쉬프트와 템포 스트레치 조작을 그대로 가져와, 이 단순한 베이스라인이 조작에 얼마나 취약한지를 반복 실험하는

중간 단계의 연구로 위치시킬 수 있습니다. SONICS 수준의 복잡한 모델과 데이터셋 사이로 가기 전에, Mel Spectrogram과 범용 CNN으로 어디까지 버틸 수 있는가를 정리하는 역할이라고 볼 수 있습니다.

---

# 2. 방법론 분석 (Methodology)

## 2.1 문제 정의

저자가 정의하는 문제는 다음과 같은 이진 분류 과제입니다.

- 입력
    - 길이 10초의 음악 오디오 클립
    - FakeMusicCaps 기반 Mel Spectrogram 이미지
- 출력
    - 레이블 0: 사람 연주 음악
    - 레이블 1: TTM 기반 딥페이크 음악

이를 수식으로 쓰면, Mel Spectrogram을 $x ∈ ㄱR^{C x H x W}$
 
---
아래 번호는 Sunday 논문 참고문헌과 정합성을 최대한 맞추어 정리했습니다.

[1] Comanducci, L., Bestagini, P., and Tubaro, S. (2024). FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models. arXiv:2409.10684.

[2] Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., and others. (2023). MusicLM: Generating Music from Text. arXiv:2301.11325.

[3] Rahman, M. A., Hakim, Z. I. A., Sarker, N. H., Paul, B., and Fattah, S. A. (2024). SONICS: Synthetic Or Not – Identifying Counterfeit Songs. arXiv:2408.14080.

[4] Afchar, D., Meseguer-Brocal, G., and Hennequin, R. (2024). Detecting music deepfakes is easy but actually hard. arXiv:2405.04181.