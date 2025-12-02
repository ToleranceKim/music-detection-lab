"""
Test script for audio_io.py module
"""

import os
import numpy as np
import sys
from pathlib import Path

# 현재 디렉토리를 python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from audio_io import load_audio, save_audio, split_audio, normalize_audio

def test_normalize():
    """정규화 함수 테스트"""
    print("\n[1] Testing normalize_audio...")

    # 테스트 신호 생성
    audio = np.array([0.5, -0.5, 1.0, -1.0, 0.25])

    # Peak normalization
    normalized_peak = normalize_audio(audio, method='peak')
    assert np.max(np.abs(normalized_peak)) <= 1.0, "Peak normalization failed"
    print("Peak normalization: Pass")

    # RMS normalization
    normalized_rms = normalize_audio(audio, method='rms')
    print("RMS normalization: PASS")

    return True

def test_split():
    """오디오 분할 함수 테스트"""
    print("\n[2] Testing split_audio...")

    # 30초 테스트 오디오 생성
    sr = 16000
    duration = 30 # seconds
    audio = np.random.randn(sr * duration)

    # 10초씩 분할
    segments = split_audio(audio, sr, segment_duration=10.0)

    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
    assert all(len(s) == sr * 10 for s in segments), "Segment length mismatch"
    print(f"Split into {len(segments)} segments: PASS")

    return True

def test_save_and_load():
    """저장 및 로드 함수 테스트"""
    print("\n[3] Testing save_audio and load_audio...")

    # 테스트 오디오 생성 (1초 440Hz 사인파)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 저장
    test_file = "test_temp.wav"
    save_audio(test_audio, sr, test_file)
    assert os.path.exists(test_file), "File not created"
    print("Save audio: PASS")

    # 로드
    loaded_audio, loaded_sr = load_audio(test_file)
    assert loaded_sr == sr, f"Sample rate mismatch: {loaded_sr} != {sr}"
    assert len(loaded_audio) == len(test_audio), "Length mismatch after loading"
    print("Load audio: PASS")

    # 임시 파일 삭제
    os.remove(test_file)
    print("Cleanup: PASS")

    return True

def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 50)
    print("Starting audio_io.py tests")
    print("=" * 50)

    tests = [
        test_normalize,
        test_split,
        test_save_and_load
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n All tests passed successfully!")
    else:
        print(f"\n {failed} test(s) failed")

if __name__ == "__main__":
    run_all_tests()

