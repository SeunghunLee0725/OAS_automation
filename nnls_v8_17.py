"""
NNLS 기반 플라즈마 UV-Vis 흡광 분석 v8.17
분석 종: O3, NO, NO2, NO3, N2O4, N2O5, HONO, HONO2

=== 주요 파라미터 (코드 하단 main 블록에서 설정) ===
  saturation_threshold    : 포화 제외 기준 흡광도 (기본 2.5)
  exclude_wavelength_range: 파장 범위 고정 제외, 예: (237, 273) / None=비활성
  hono_no2_factor         : HONO 상한 인자 k (기본 0.01, 범위 0.003~0.05)
                            HONO_max = [NO2] × k × RH  (Stutz 2004)
  humidity_weight         : RH/100, 0=건조 (실행 시 입력)
  n2o5_scale              : N2O5 스케일 (기본 1.0)
                            200nm 초반에서 N2O5 크로스섹션이 O3보다 강하므로
                            n2o5_scale < 1.0으로 과분석 억제 가능
  apply_n2o5_stoich_cap   : N2O5 화학량론 상한 적용 여부 (기본 False)
                            True : N2O5 ≤ min([NO2],[NO3]) 강제 적용
                            False: NNLS 피팅값 그대로 사용 (권장)

=== 버전 이력 ===
v8.17 n2o5_scale 버그 수정:
      O3-NO 규칙 발동 시(O3>=2e14 and NO>0) else 분기에서
      N2O5가 scale 없이 자유 재피팅되어 n2o5_scale 설정이 무시되던 문제 수정.
      수정 후: O3-NO 규칙 발동 여부와 무관하게 n2o5_scale이 항상 적용됨.
v8.16 N2O5 stoich cap 옵션 추가 (apply_n2o5_stoich_cap)
v8.15 포화 처리 개선: threshold 기본값 2.75→2.5,
      exclude_wavelength_range 파라미터 추가 (파장 범위 고정 제외)
v8.14 NO2 Vis 제약 추가, O3-NO 규칙 복원, N2O5 상한 교체,
      step3_red NO2 제외, Step1에서 HONO 제외 (N2O5 과소추정 방지)
v8.13 N2O4 Step1 이동, humidity_weight 초기값 버그 수정
v8.12 이론-코드 정합성 개선 (HONO 반응식, 온도 연동, NO3 귀속, O3-NO 규칙)
v8.11 HONO/HONO2 전구체 기반 상한 적용, RH% 직접 입력
v8.10 NO3 안정화 (피크 제약, 시계열 제약)
v8.8  온도 프로파일 CSV 지원
v8    Detection Limit 추가
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import os
import re
from typing import Dict, List, Tuple, Optional
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')
class BatchPlasmaAnalyzer:
    """
    플라즈마 배치 분석 시스템 (습도 조절 가능) v8.17
    """

    # === MODIFIED: 파일명 시간 토큰 정규식 (공백 허용, 어디에 있어도 OK) ===
    TIME_TOKEN = re.compile(r'__\s*(\d+)\s*__')

    def __init__(self, cross_section_path: str, path_length: float = 5.0,
                 correct_overfit: bool = True,
                 correct_n2o4_equilibrium: bool = True, temperature: float = 295.0,
                 noise_threshold: float = 1e-4,
                 no3_peak_constraint: bool = True, no3_peak_range: tuple = (615, 635),
                 no3_damping_factor: float = 0.90, no3_min_threshold: float = 0.005,
                 no3_max_iterations: int = 150,
                 no3_window_nm: float = 3.0,
                 uv_start: float = 205.0,
                 n2o5_scale: float = 0.5,
                 no2_vis_constraint: bool = True,
                 no2_vis_range: tuple = (350, 450),
                 no2_vis_window_nm: float = 10.0,
                 no2_vis_noise_threshold: float = 5e-4,
                 no2_vis_tolerance: float = 0.10,
                 no2_vis_percentile: float = 5.0,
                 hono_no2_factor: float = 0.01,
                 apply_n2o5_stoich_cap: bool = False):
        """
        초기화

        Parameters:
        -----------
        cross_section_path : str
            Cross section 데이터 경로
        path_length : float
            흡광 경로 길이 (cm)
        correct_overfit : bool
            Overfitting 보정 활성화 (Step 4)
            - True: NO2 영역에서 overfitting 시 HONO 조정
            - False: 보정 없이 피팅 결과 그대로 사용
        correct_n2o4_equilibrium : bool
            N2O4 평형 보정 활성화
            - True: NO2-N2O4 평형 관계를 이용해 N2O4 농도 보정
            - False: NNLS 피팅 결과 그대로 사용
        temperature : float
            온도 (K) - N2O4 평형 상수 계산에 사용
        noise_threshold : float
            Absorbance 노이즈 임계값 - 이 값보다 낮은 absorbance는 0으로 처리
            기본값: 1e-4 (0.0001)
        no3_peak_constraint : bool
            NO3 피크 영역 강제 제약 활성화 (기본값: True)
            - True: 피크 영역에서 fitted ≤ measured 강제 (R² 감소 허용)
            - False: 제약 없이 일반 NNLS
        no3_peak_range : tuple
            NO3 피크 영역 범위 (nm) (기본값: (616, 635))
        no3_damping_factor : float
            NO3 감소 비율 (기본값: 0.95 = 5% 감소)
            - 0.90: 10% 감소 (더 빠름)
            - 0.95: 5% 감소 (권장)
            - 0.98: 2% 감소 (더 천천히)
        no3_min_threshold : float
            NO3 최소 농도 임계값 (초기값 대비 비율) (기본값: 0.005 = 0.5%)
            - 이 값보다 작아지면 iteration 중단
            - 0.001: 0.1% (매우 공격적)
            - 0.005: 0.5% (권장)
            - 0.01: 1.0% (보수적)
        no3_max_iterations : int
            최대 iteration 횟수 (기본값: 150)
            - 50: 빠른 종료
            - 150: 권장
            - 300: 완전한 제거 추구
        no3_window_nm : float
            경향 비교용 슬라이딩 윈도우 크기 (nm) (기본값: 3.0)  [v8.9]
            - 포인트별 비교 대신 window_nm 범위 평균으로 NO3 제약
            - 작을수록 포인트 비교에 가까워짐 (노이즈 민감)
            - 클수록 넓은 경향 반영 (노이즈에 강건)
            - 권장: 2.0~5.0 nm
        uv_start : float
            UV 피팅 시작 파장 (nm) (기본값: 200.0)
            - 200nm 초반 영역 피팅 개선을 위해 200nm부터 시작 권장
            - 주의: NO는 210nm부터 cross section 있음
        n2o5_scale : float
            N2O5 스케일 조절 (기본값: 1.0)
            - 200nm 초반 피팅 개선을 위한 파라미터
            - 1.0: 제한 없음 (기존 방식) ← 기본값
            - 0.25: N2O5 25%로 제한
            - 0.0: N2O5 제외 - 200nm 초과 최소화, O3 정확도 유지
        no2_vis_constraint : bool
            350-450nm NO2 Overfit 제약 활성화 (기본값: True)  [v8.14]
            - True: 350-450nm에서 sliding window 평균으로 NO2 상한 제약
            - False: 제약 없음 (기존 방식)
        no2_vis_range : tuple
            NO2 Vis 제약 구간 (nm) (기본값: (350, 450))  [v8.14]
        no2_vis_window_nm : float
            NO2 Vis 제약용 슬라이딩 윈도우 크기 (nm) (기본값: 10.0)  [v8.14]
            - 노이즈 포인트 무시, 경향 기반 비교
            - 작을수록 포인트 비교에 가까움 (노이즈 민감)
            - 클수록 넓은 경향 반영 (노이즈에 강건)
        hono_no2_factor : float
            HONO 상한 계산용 [HONO]/[NO2] 비율 인자 (기본값: 0.01)  [v8.14]
            HONO_max = [NO2] × hono_no2_factor × (RH/100)
            근거: Stutz et al. (2004) [HONO]/[NO2] ∝ RH,
                  실험적 비율 k = 0.003~0.05 (표면·환경 의존)
            - 0.003: 하한 (매우 보수적)
            - 0.01 : 기본값 (보수적 추정)
            - 0.05 : 상한 (습한 표면 환경)
        apply_n2o5_stoich_cap : bool
            N2O5 화학량론적 상한 적용 여부 (기본값: False)  [v8.16]
            - False: NNLS 피팅값 그대로 사용 (권장)
                     200nm 초반에서 N2O5 크로스섹션이 O3보다 강하므로
                     NNLS가 실제 N2O5 신호를 정량할 수 있음
            - True : N2O5 ≤ min([NO2],[NO3]) 강제 적용 (v8.14 방식, 비권장)
                     피팅 후 잔존 농도로 상한을 정하는 것은 이론적 근거 부족
        """
        self.cross_section_path = cross_section_path
        self.path_length = path_length

        # === 피팅 옵션 ===
        self.correct_overfit = correct_overfit
        self.correct_n2o4_equilibrium = correct_n2o4_equilibrium
        self.temperature = temperature
        self.noise_threshold = noise_threshold
        
        # === NO3 피크 제약 옵션 ===
        self.no3_peak_constraint = no3_peak_constraint
        self.no3_peak_range = no3_peak_range
        self.no3_damping_factor = no3_damping_factor
        self.no3_min_threshold = no3_min_threshold
        self.no3_max_iterations = no3_max_iterations
        self.no3_window_nm = no3_window_nm          # [v8.9] 경향 기반 윈도우 크기
        
        # === UV 피팅 시작점 (v8.5 추가) ===
        self.uv_start = uv_start
        
        # === N2O5 스케일 조절 (v8.7 추가) ===
        # 200nm 초반 피팅 개선을 위해 N2O5 기여도 제한
        # 0.0: N2O5 제외 (권장 - 200nm 초과 최소화)
        # 0.25: N2O5 25%로 제한
        # 1.0: 제한 없음 (기존 방식)
        self.n2o5_scale = n2o5_scale

        # === NO2 Vis 제약 (v8.14) ===
        self.no2_vis_constraint = no2_vis_constraint
        self.no2_vis_range = no2_vis_range
        self.no2_vis_window_nm = no2_vis_window_nm
        self.no2_vis_noise_threshold = no2_vis_noise_threshold
        self.no2_vis_tolerance = no2_vis_tolerance
        self.no2_vis_percentile = no2_vis_percentile

        # === HONO 상한 인자 (v8.14) ===
        self.hono_no2_factor = hono_no2_factor

        # === N2O5 stoich cap (v8.16) ===
        self.apply_n2o5_stoich_cap = apply_n2o5_stoich_cap

        # 화학종 순서 (HONO, HONO2 포함)
        self.species_order = ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2']
        self.species_list = []
        self.cross_sections = {}
        self.fitting_range = (205, 650)  # nm
        
        # 습도 관련
        self.humidity_weight = 0   # [v8.13 fix] 기본값 0 (건조 조건); get_humidity_weight() 호출 시 갱신

        # === 포화 영역 제외 설정 ===
        # [v8.15] 두 가지 방법 지원
        #
        # 방법 A: threshold 기반 (기본값 2.5)
        #   absorbance >= saturation_threshold 인 포인트 제외
        #   데이터마다 제외 구간이 자동으로 달라짐
        #   이전 기본값 2.75 → 2.5로 변경
        #   근거: 이 데이터셋(165037)의 최대 흡광도가 3.2~3.5로
        #         2.75~3.5 구간도 포화 왜곡이 심함
        #
        # 방법 B: 파장 범위 기반 (exclude_wavelength_range)
        #   지정한 파장 구간을 모든 파일에서 고정 제외
        #   시계열 비교 시 일관성 확보에 유리
        #   권고 범위: (237, 273) nm
        #   근거: 165037 데이터셋 전체에서 공통 포화 구간
        #
        # 두 방법 동시 적용 가능 (AND 조건)
        self.saturation_threshold = 2.75          # 방법 A
        self.exclude_wavelength_range = None     # 방법 B: None 또는 (wl_min, wl_max)
    

        # === Detection Limits (개선된 추정값 - 논문 기반) ===
        # 기준: Huh et al. 2024 (Plasma Sources Sci. Technol. 33 075007)
        # 실험 조건:
        #   - Path length: 5 cm (논문: 15 cm)
        #   - Integration: 100ms × 20 = 2000ms (논문: 20ms × 25 = 500ms)
        #   - SNR improvement: 2.0x (더 긴 적분 시간)
        # 
        # 계산: 논문 DL × (15cm/5cm) / SNR_improvement
        #     = 논문 DL × 3 / 2 = 논문 DL × 1.5
        # 
        # 이 값은 Blank 측정 수행 시 달성 가능한 현실적 추정값
        self.detection_limits = {
            'O3': 3.7e14,      # 15 ppm (논문: 10 ppm × 1.5)
            'NO': 6.6e14,      # 27 ppm (논문: 18 ppm × 1.5)
            'NO2': 2.6e14,     # 10.5 ppm (논문: 7 ppm × 1.5)
            'NO3': 5.2e13,     # 2.1 ppm (논문: 1.4 ppm × 1.5)
            'N2O4': 3.1e13,    # 1.2 ppm (논문: 0.8 ppm × 1.5)
            'N2O5': 1.1e14,    # 4.6 ppm (논문: 3.1 ppm × 1.5)
            'HONO': 6.0e13,    # 2.4 ppm (논문: 1.6 ppm × 1.5)
            'HONO2': 7.7e14    # 31.5 ppm (논문: 21 ppm × 1.5, HNO3)
        }

        # 배치 처리 결과 저장
        self.batch_results = {}

        # 화학 반응 규칙 설정
        self.setup_chemical_rules()

        # === 온도 프로파일 (v8.8) ===
        # load_temperature_profile() 호출 시 채워짐
        self.temperature_profile = None   # {'times': np.array, 'temps_K': np.array} or None

        print("\n" + "="*60)
        print("BATCH PLASMA ANALYZER - 개선된 버전 v8.18")
        print("[v8.18] 350-450nm NO2 Vis Overfit 제약 (개선판):")
        print(f"  - no2_vis_constraint: {'ON' if self.no2_vis_constraint else 'OFF'}")
        print(f"  - 제약 구간: {self.no2_vis_range[0]}-{self.no2_vis_range[1]}nm")
        print(f"  - 슬라이딩 윈도우: {self.no2_vis_window_nm}nm (노이즈 내성)")
        print(f"  - 노이즈 게이트: {self.no2_vis_noise_threshold:.1e}")
        print(f"  - 허용 오차: {self.no2_vis_tolerance:.0%}")
        print(f"  - 집계 percentile: {self.no2_vis_percentile}th")
        print("[v8.14] HONO 상한 전구체 변경: N2O4 → NO2")
        print(f"  - HONO_max = [NO2] × {self.hono_no2_factor} × RH/100  (Stutz 2004)")
        print(f"  - apply_hono_precursor_cap 순서: Step4 평형 보정 이후로 이동")
        print("[v8.13] N2O4 피팅 구간: Step2(Vis) → Step1(UV)")
        print("[v8.11] HONO/HONO2 피팅 구조 개선:")
        print("  ① Step1 UV에 HONO/HONO2 추가 (humidity>0, cross section 원본)")
        print("  ② × 0.3 임의 가중치 제거 → 전구체 기반 상한 적용")
        print("     HONO  ≤ [N2O4] × RH/100")
        print("     HONO2 ≤ [N2O5] × 2 × RH/100")
        print("  ③ Step3 보정에서 HONO2 제외 (370-400nm 흡수 없음)")
        print("  ④ 습도 입력: RH(%) 직접 입력 (Stutz et al. 2004)")
        print("[v8.10] NO3 안정화: 최대값 기준 제약 + 시간축 자동 허용 범위")
        print("[v8.8] 시간별 온도 프로파일 지원 (CSV 입력, N2O4 평형 동적 반영)")
        print("[v8.7] N2O5 스케일 조절 (200nm 피팅 개선):")
        print(f"  - N2O5 스케일: {self.n2o5_scale:.2f}")
        if self.n2o5_scale == 0.0:
            print(f"  - N2O5 제외 (O3, NO, N2O4만 피팅)")
        elif self.n2o5_scale < 1.0:
            print(f"  - N2O5 {self.n2o5_scale*100:.0f}%로 제한")
        else:
            print(f"  - N2O5 제한 없음 (자유 피팅, 권장)")
        print(f"[v8.17] n2o5_scale 버그 수정 (O3-NO 규칙 발동 시 scale 무시 문제):")
        print(f"  - O3-NO 규칙 else 분기에서도 n2o5_scale 항상 적용")
        print(f"[v8.16] N2O5 stoich cap: {'ON' if self.apply_n2o5_stoich_cap else 'OFF (권장)'}")
        if self.apply_n2o5_stoich_cap:
            print(f"  - N2O5 ≤ min([NO2],[NO3]) 적용")
        else:
            print(f"  - NNLS 피팅값 그대로 사용")
        print(f"[v8.5] UV 피팅 시작점: {self.uv_start}nm")
        print("[v8.10] NO3 최대값 기준 제약 (peak region max cap):")
        print(f"  - 피크 영역: {self.no3_peak_range[0]}-{self.no3_peak_range[1]}nm")
        print(f"  - max(NO3 기여) ≤ max(available) → 1회 직접 cap")
        print(f"  - Damping: {self.no3_damping_factor} (각 iter {(1-self.no3_damping_factor)*100:.0f}% 감소)")
        print("[v8] 2단계 피팅 (확장) + N2O4 평형 보정 + Detection Limit:")
        print(f"  1단계: UV({self.uv_start:.0f}-350nm) → O3, NO, N2O5")
        print("  2단계: Vis(350-650nm) → NO2, N2O4, NO3 (+HONO)")
        print(f"Overfitting 보정: {'ON' if self.correct_overfit else 'OFF'}")
        print(f"N2O4 평형 보정: {'ON' if self.correct_n2o4_equilibrium else 'OFF'}")
        print(f"NO3 피크 제약: {'ON' if self.no3_peak_constraint else 'OFF'}")
        print(f"온도: {self.temperature:.1f} K ({self.temperature-273:.1f}°C)")
        print(f"노이즈 임계값: {self.noise_threshold:.0e}")
        print(f"포화 영역 제외: absorbance >= {self.saturation_threshold}")
        print(f"Detection Limit 적용: ON (일반 UV-Vis 기준)")
        print("="*60)

    def setup_chemical_rules(self):
        """
        화학 반응 규칙 설정
        """
        # 1차 생성물 (플라즈마에서 직접 생성)
        self.primary_species = ['O3', 'NO', 'NO2', 'NO3']

        # 2차 생성물과 전구체 관계
        self.secondary_reactions = {
            'N2O5': {
                'reactants': ['NO2', 'NO3'],
                'type': 'equilibrium',
                'max_ratio': 0.5,
                'min_ratio': 0.001,
                'requires_O3': True,
                'description': 'NO2 + NO3 <-> N2O5'
            },
            'N2O4': {
                'reactants': ['NO2'],
                'type': 'equilibrium',
                'max_ratio': 0.3,
                'min_ratio': 0.001,
                'requires_O3': False,
                'description': '2 NO2 <-> N2O4'
            },
            'HONO': {
                'reactants': ['NO2', 'N2O4'],
                'type': 'hydrolysis',
                'max_ratio': 0.5,
                'min_ratio': 0.01,
                'requires_O3': False,
                'description': 'N2O4 + H2O -> HONO + HNO3'
            },
            'HONO2': {
                'reactants': ['N2O5', 'N2O4'],
                'type': 'hydrolysis',
                'max_ratio': 1.0,
                'min_ratio': 0.01,
                'requires_O3': False,
                'description': 'N2O5 + H2O -> 2 HONO2'
            }
        }

        # 경쟁 반응 및 Selection Rules
        self.competitive_reactions = {
            'NO_O3': {
                'reactants': ['NO', 'O3'],
                'products': ['NO2'],
                'rate': 'very_fast',
                'description': 'NO + O3 -> NO2 + O2',
                'rule': 'O3 >= 2e14 → NO = 0 (동시 존재 불가) [v8.14: v8.7 방식 복원]'
            },
            'NO2_O3': {
                'reactants': ['NO2', 'O3'],
                'products': ['NO3'],
                'rate': 'moderate',
                'description': 'NO2 + O3 -> NO3 + O2'
            }
        }

        print(f"\n[v8.5] 2단계 피팅 알고리즘 (확장) + N2O4 평형 보정:")
        print(f"  1단계: UV({self.uv_start:.0f}-350nm) → O3, NO, N2O5")
        print("  2단계: Vis(350-650nm) → NO2, N2O4, NO3 (+HONO)")
        print("  * O3-NO 규칙: O3 >= 2e14 → NO 제거 [v8.14: v8.7 방식 복원]")
        print("  * HONO/HONO2: 습도 > 0일 때만 2단계에서 함께 피팅")
        print(f"  * 포화 영역 (absorbance >= {self.saturation_threshold}) 자동 제외")
        print("  * Detection Limit 이하 농도는 0으로 처리됨")
        if self.correct_n2o4_equilibrium:
            print(f"  * N2O4 평형 보정: ON (T={self.temperature:.1f} K)")
    
    # =========================================================
    # [v8.8] 시간별 온도 프로파일 메서드
    # =========================================================

    def load_temperature_profile(self, csv_path: str,
                                 time_col: str = None,
                                 temp_col: str = None) -> bool:
        """
        온도 CSV 파일 로드 및 선형 보간 준비

        CSV 형식: time(s), temperature(C)  (컬럼명 자동 감지)
        내부 저장: Kelvin 변환 후 self.temperature_profile에 보관

        Parameters
        ----------
        csv_path : str
            온도 CSV/TXT 파일 경로
        time_col : str, optional
            시간 컬럼명 (None이면 자동 감지)
        temp_col : str, optional
            온도 컬럼명 (None이면 자동 감지)

        Returns
        -------
        bool
            성공 여부
        """
        # --- 컬럼 자동 감지 후보 ---
        TIME_CANDIDATES = ['time', 'Time', 'TIME', 't', 'sec', 'second', 'seconds',
                           'time(s)', 'Time(s)', 't(s)', 'time_s']
        TEMP_CANDIDATES = ['temperature', 'Temperature', 'TEMPERATURE',
                           'temp', 'Temp', 'TEMP',
                           'temperature(c)', 'Temperature(C)', 'temperature(C)',
                           'temperature(k)', 'Temperature(K)',
                           'temp_c', 'temp_C', 'T', 'T_C', 'T_K']

        try:
            # 구분자 자동 판별 (csv / tsv / 공백)
            ext = os.path.splitext(csv_path)[1].lower()
            if ext in ('.tsv', '.txt'):
                df = pd.read_csv(csv_path, sep=None, engine='python')
            else:
                df = pd.read_csv(csv_path)

            cols = list(df.columns)
            print(f"\n[온도 CSV] 로드: {os.path.basename(csv_path)}")
            print(f"  컬럼 목록: {cols}")

            # --- 시간 컬럼 ---
            if time_col is None:
                for cand in TIME_CANDIDATES:
                    if cand in cols:
                        time_col = cand
                        break
                # 그래도 없으면 첫 번째 컬럼
                if time_col is None:
                    time_col = cols[0]
                    print(f"  ⚠ 시간 컬럼 자동 감지 실패 → 첫 번째 컬럼 사용: '{time_col}'")
                else:
                    print(f"  ✓ 시간 컬럼 자동 감지: '{time_col}'")
            else:
                print(f"  ✓ 시간 컬럼 지정: '{time_col}'")

            # --- 온도 컬럼 ---
            if temp_col is None:
                for cand in TEMP_CANDIDATES:
                    if cand in cols:
                        temp_col = cand
                        break
                if temp_col is None:
                    # 시간 컬럼 제외 후 두 번째 컬럼
                    remaining = [c for c in cols if c != time_col]
                    temp_col = remaining[0] if remaining else cols[1]
                    print(f"  ⚠ 온도 컬럼 자동 감지 실패 → 컬럼 사용: '{temp_col}'")
                else:
                    print(f"  ✓ 온도 컬럼 자동 감지: '{temp_col}'")
            else:
                print(f"  ✓ 온도 컬럼 지정: '{temp_col}'")

            times = df[time_col].values.astype(float)
            temps_raw = df[temp_col].values.astype(float)

            # 섭씨 → 켈빈 변환 (최솟값이 100K 미만이면 섭씨로 간주)
            if np.min(temps_raw) < 100:
                temps_K = temps_raw + 273.15
                unit_str = "°C → K 변환"
            else:
                temps_K = temps_raw
                unit_str = "K (변환 없음)"

            print(f"  온도 단위: {unit_str}")
            print(f"  시간 범위: {times[0]:.1f} ~ {times[-1]:.1f} s ({len(times)}개 포인트)")
            print(f"  온도 범위: {np.min(temps_K)-273.15:.1f} ~ {np.max(temps_K)-273.15:.1f} °C"
                  f"  ({np.min(temps_K):.1f} ~ {np.max(temps_K):.1f} K)")

            # 시간 오름차순 정렬 (보간 필수)
            sort_idx = np.argsort(times)
            self.temperature_profile = {
                'times': times[sort_idx],
                'temps_K': temps_K[sort_idx],
                'source': os.path.basename(csv_path)
            }
            print(f"  ✓ 온도 프로파일 로드 완료")
            return True

        except Exception as e:
            print(f"  ✗ 온도 CSV 로드 실패: {e}")
            self.temperature_profile = None
            return False

    def get_temperature_at_time(self, t: float) -> float:
        """
        특정 시간 t(s)에서의 온도(K) 반환 (선형 보간, 범위 외 클램핑)

        온도 프로파일이 없으면 self.temperature(초기 고정값)를 반환.
        """
        if self.temperature_profile is None:
            return self.temperature

        times = self.temperature_profile['times']
        temps_K = self.temperature_profile['temps_K']

        # np.interp: 범위 밖은 양 끝값으로 클램핑
        T = float(np.interp(t, times, temps_K))
        return T

    def get_temperature_profile(self):
        """
        [인터랙티브] 온도 CSV 경로 입력 받기 (배치 분석 전 호출)

        사용자가 Enter만 누르면 고정 온도(self.temperature)를 유지.
        """
        print("\n" + "="*60)
        print("[v8.8] 시간별 온도 프로파일 설정")
        print(f"  현재 고정 온도: {self.temperature:.1f} K ({self.temperature-273.15:.1f} °C)")
        print("  온도 CSV 형식: time(s), temperature(C)")
        print("  Enter만 누르면 고정 온도 유지")
        print("="*60)

        while True:
            csv_path = input("온도 CSV 파일 경로 (Enter = 고정 온도 사용): ").strip()
            if csv_path == "":
                print(f"  → 고정 온도 사용: {self.temperature:.1f} K")
                return
            if not os.path.exists(csv_path):
                print(f"  ✗ 파일 없음: {csv_path}")
                continue
            ok = self.load_temperature_profile(csv_path)
            if ok:
                return
            else:
                retry = input("  다시 입력하시겠습니까? (y/n): ").strip().lower()
                if retry != 'y':
                    print(f"  → 고정 온도 사용: {self.temperature:.1f} K")
                    return

    def calculate_n2o4_equilibrium_constant(self, T: float) -> float:
        """
        NO2-N2O4 평형 상수 계산 (온도 의존성)
        
        2 NO2 <-> N2O4
        Keq = [N2O4] / [NO2]^2
        
        Parameters:
        -----------
        T : float
            온도 (K)
            
        Returns:
        --------
        Keq : float
            평형 상수 (cm³/molecule)
            
        Reference:
        ----------
        Atkinson, R. et al., "Evaluated kinetic and photochemical data for 
        atmospheric chemistry: Volume I - gas phase reactions of Ox, HOx, 
        NOx and SOx species", Atmos. Chem. Phys. 4 (2004) 1461-1738
        DOI: 10.5194/acp-4-1461-2004
        
        Keq = k_forward / k_reverse = 8.7e-29 * exp(6460/T) cm³/molecule
        """
        Keq = 8.7e-29 * np.exp(6460.0 / T)
        return Keq

    def calculate_n2o5_equilibrium_constant(self, T: float) -> float:
        """
        NO2 + NO3 → N2O5 평형 상수 계산 (온도 의존성)

        NO2 + NO3 <-> N2O5
        Keq = [N2O5] / ([NO2] × [NO3])

        Parameters
        ----------
        T : float
            온도 (K)

        Returns
        -------
        Keq : float
            평형 상수 (cm³/molecule)

        Reference
        ---------
        IUPAC Task Group 2009 (Atkinson et al.):
        Keq = k_forward / k_reverse
            = 3.0e-27 × exp(10991/T)  cm³/molecule
        유효 범위: 200–330 K
        """
        Keq = 3.0e-27 * np.exp(10991.0 / T)
        return Keq

    def correct_n2o5_by_equilibrium(self, conc: np.ndarray, fitting_info: dict) -> Tuple[np.ndarray, dict]:
        """
        N2O5 농도 화학량론적 상한 적용  [v8.14: 상한 계산식 교체]

        [v8.9 문제]
          기존: N2O5_max = Keq(T) × [NO2] × [NO3]
          Keq(295K) ≈ 4.5e-11 cm³/molecule → 상한이 수천만 ppm → dead code

          근본 원인:
            Keq가 크다 = 평형이 N2O5 방향으로 강하게 치우침
            ≠ N2O5가 많이 생성될 수 있음
            Keq × [NO2] × [NO3]는 열역학적 평형 농도이지
            현실적 생성 상한이 아님

        [v8.14 수정]
          화학량론적 상한으로 교체:
            N2O5 ≤ min([NO2], [NO3])
          근거: NO2 + NO3 → N2O5  (1:1 몰비)
            전구체 중 적은 쪽 이상의 N2O5는 생성 불가
            온도 독립적, 명확한 물리적 근거
          참고: Atkinson et al., ACP 4 (2004) 1461-1738
        """
        idx_map = {sp: self.species_list.index(sp) for sp in self.species_list}
        no2_idx  = idx_map.get('NO2')
        no3_idx  = idx_map.get('NO3')
        n2o5_idx = idx_map.get('N2O5')

        if any(i is None for i in [no2_idx, no3_idx, n2o5_idx]):
            return conc, fitting_info

        no2_conc    = conc[no2_idx]
        no3_conc    = conc[no3_idx]
        n2o5_fitted = conc[n2o5_idx]

        # NO2 또는 N2O5가 0이면 보정 불필요
        # [v8.14] NO3=0 → N2O5=0 강제 로직 제거
        #   근거: DL 이하 ≠ 실제 없음, N2O5 열분해 반감기(~36s@295K) >> 측정 시간
        if no2_conc <= 0 or n2o5_fitted <= 0:
            fitting_info['n2o5_equilibrium'] = {
                'correction_applied': False,
                'description': 'No correction needed (NO2 or N2O5 is zero)'
            }
            return conc, fitting_info

        # [v8.14] 화학량론적 상한: N2O5 ≤ min([NO2], [NO3])
        # NO3=0이어도 N2O5가 존재할 수 있으므로(열분해 잔존),
        # NO3=0인 경우에는 화학량론 상한 적용을 건너뜀
        if no3_conc <= 0:
            fitting_info['n2o5_equilibrium'] = {
                'correction_applied': False,
                'description': 'Stoichiometric cap skipped (NO3=0, N2O5 may persist from decomposition)'
            }
            return conc, fitting_info

        # [v8.16] apply_n2o5_stoich_cap=False이면 상한 적용 안 함
        if not self.apply_n2o5_stoich_cap:
            fitting_info['n2o5_equilibrium'] = {
                'correction_applied': False,
                'n2o5_fitted': n2o5_fitted,
                'description': 'Stoich cap disabled (apply_n2o5_stoich_cap=False) [v8.16]'
            }
            return conc, fitting_info

        n2o5_max = min(no2_conc, no3_conc)   # 화학량론적 상한

        conc_corrected = conc.copy()

        if n2o5_fitted > n2o5_max:
            conc_corrected[n2o5_idx] = n2o5_max
            fitting_info['n2o5_equilibrium'] = {
                'correction_applied': True,
                'n2o5_fitted': n2o5_fitted,
                'n2o5_max': n2o5_max,
                'n2o5_corrected': n2o5_max,
                'ratio_before': n2o5_fitted / n2o5_max,
                'no2_concentration': no2_conc,
                'no3_concentration': no3_conc,
                'description': (f'N2O5 stoich cap: {n2o5_fitted:.2e} → {n2o5_max:.2e} '
                                f'(={min.__name__}([NO2],[NO3]), '
                                f'was {n2o5_fitted/n2o5_max:.2f}x over) [v8.14]')
            }
        else:
            fitting_info['n2o5_equilibrium'] = {
                'correction_applied': False,
                'n2o5_fitted': n2o5_fitted,
                'n2o5_max': n2o5_max,
                'ratio': n2o5_fitted / n2o5_max,
                'no2_concentration': no2_conc,
                'no3_concentration': no3_conc,
                'description': (f'N2O5 within stoich limit '
                                f'({n2o5_fitted/n2o5_max:.1%} of min([NO2],[NO3])) [v8.14]')
            }

        return conc_corrected, fitting_info
    
    def apply_no3_peak_constraint(self, wavelengths: np.ndarray, b: np.ndarray,
                                  A: np.ndarray, conc: np.ndarray, no3_idx: int) -> np.ndarray:
        """
        NO3 기여도 제약 적용 (v8.10: 영역 최대값 기준)

        [v8.10 변경]
        기존(v8.9): 슬라이딩 윈도우 평균(경향)으로 위반 판정
        변경(v8.10): 피크 영역에서 NO3 기여도의 최대값이
                     measured 최대값을 넘지 않도록 제약
                     → 노이즈성 0값 포인트 영향 완전 차단
                     → 실제 피크 높이 기준으로 상한 결정

        제약 조건:
            max(NO3_contribution, peak_region) ≤ max(available, peak_region)
        """
        peak_region = (wavelengths >= self.no3_peak_range[0]) & \
                      (wavelengths <= self.no3_peak_range[1])

        print(f"    [DEBUG] apply_no3_peak_constraint 시작 (최대값 기준 v8.10)")
        print(f"    [DEBUG] 피크 영역: {self.no3_peak_range[0]}-{self.no3_peak_range[1]}nm")
        print(f"    [DEBUG] 피크 영역 포인트: {np.sum(peak_region)}")

        if not np.any(peak_region):
            print(f"    [DEBUG] 피크 영역 없음 - 제약 스킵")
            return conc

        initial_no3 = conc[no3_idx]
        print(f"    [DEBUG] 초기 NO3: {initial_no3:.2e}")

        if initial_no3 == 0:
            print(f"    [DEBUG] NO3가 0 - 제약 스킵")
            return conc

        # ── 다른 화학종 기여도 계산 ──────────────────────────────
        other_conc = conc.copy()
        other_conc[no3_idx] = 0
        fitted_others      = A @ other_conc
        fitted_others_peak = fitted_others[peak_region]
        measured_peak      = b[peak_region]
        no3_cs_peak        = A[peak_region, no3_idx]

        # NO3가 사용 가능한 공간 = measured - others
        available_peak = measured_peak - fitted_others_peak

        # ── 최대값 기준 상한 계산 ────────────────────────────────
        # available의 최대값 → NO3 기여 최대값이 이를 넘지 않도록
        available_max = np.max(available_peak)
        no3_contrib_max_per_unit = np.max(no3_cs_peak)  # 단위 농도당 최대 기여

        if no3_contrib_max_per_unit <= 0 or available_max <= 0:
            print(f"    [DEBUG] NO3 cross section 또는 available이 0 → 스킵")
            return conc

        # 최대값 기준 NO3 상한 농도
        no3_max_by_peak = available_max / no3_contrib_max_per_unit

        print(f"\n    [NO3 최대값 기준 제약] {self.no3_peak_range[0]}-{self.no3_peak_range[1]}nm")
        print(f"      초기 NO3: {initial_no3:.2e}")
        print(f"      available 최대값: {available_max:.5f}")
        print(f"      NO3 상한 농도:    {no3_max_by_peak:.2e}")

        conc_result = conc.copy()

        if initial_no3 <= no3_max_by_peak:
            print(f"      ✓ 이미 만족 (NO3 = {initial_no3:.2e} ≤ 상한 {no3_max_by_peak:.2e})")
            return conc_result

        # 상한 초과 시 직접 cap (iterative 불필요)
        conc_result[no3_idx] = no3_max_by_peak
        reduction_pct = (1 - no3_max_by_peak / initial_no3) * 100
        print(f"      → NO3 cap 적용: {initial_no3:.2e} → {no3_max_by_peak:.2e} "
              f"({reduction_pct:.1f}% 감소)")

        return conc_result
    
    def apply_no2_vis_constraint(self, wavelengths: np.ndarray, b: np.ndarray,
                                 A: np.ndarray, conc: np.ndarray,
                                 no2_idx: int) -> Tuple[np.ndarray, dict]:
        """
        [v8.18] 350-450nm NO2 Vis Overfit 제약 (개선판)

        목표: total fitted absorbance가 measured를 초과하지 않도록 NO2 농도를 제어.

        v8.14 대비 개선사항:
          1. 노이즈 게이트: 측정 신호가 노이즈 수준이면 제약 비활성화
          2. 전체 fitted vs measured 비교 (available 계산 오차 제거)
          3. 허용 오차(tolerance) 추가: 소폭 초과는 무시
          4. 5th-percentile 집계: 단일 최악 윈도우 대신 robust 통계 사용
        """
        info = {}
        conc_result = conc.copy()

        if not self.no2_vis_constraint:
            info['applied'] = False
            info['description'] = 'no2_vis_constraint = OFF'
            return conc_result, info

        # 제약 구간 마스크
        region = (wavelengths >= self.no2_vis_range[0]) & \
                 (wavelengths <= self.no2_vis_range[1])

        if not np.any(region) or conc[no2_idx] <= 0:
            info['applied'] = False
            info['description'] = 'No points in region or NO2=0'
            return conc_result, info

        # ── 노이즈 게이트 ──────────────────────────────────────────────────────
        # 측정 신호의 75th percentile이 noise_threshold 미만이면 제약 건너뜀
        signal_level = np.percentile(b[region], 75)
        if signal_level < self.no2_vis_noise_threshold:
            info['applied'] = False
            info['signal_level'] = signal_level
            info['noise_threshold'] = self.no2_vis_noise_threshold
            info['description'] = (f'NO2 vis constraint: 노이즈 게이트 통과 실패 '
                                   f'(signal_75pct={signal_level:.2e} < '
                                   f'threshold={self.no2_vis_noise_threshold:.2e})')
            return conc_result, info

        wl_r = wavelengths[region]
        b_r = b[region]

        # 전체 fitted absorbance (현재 conc 기준)
        fitted_total = (A @ conc)[region]

        # NO2 cross section 기여 (단위 농도당)
        no2_cs = A[region, no2_idx]  # 이미 × path_length

        # ── 슬라이딩 윈도우 스캔 ───────────────────────────────────────────────
        hw = self.no2_vis_window_nm / 2.0
        scale_factors = []  # 위반 윈도우의 scale factor 수집

        for i, wl in enumerate(wl_r):
            win = (wl_r >= wl - hw) & (wl_r <= wl + hw)
            if not np.any(win):
                continue

            measured_win = b_r[win].mean()
            fitted_win = fitted_total[win].mean()
            no2_cs_win = no2_cs[win].mean()

            # 노이즈 수준 윈도우는 건너뜀
            if measured_win < self.no2_vis_noise_threshold or no2_cs_win <= 0:
                continue

            # 허용 오차 포함 위반 검사: fitted > measured × (1 + tolerance)
            if fitted_win > measured_win * (1.0 + self.no2_vis_tolerance):
                # NO2를 얼마나 줄여야 하는가?
                # fitted_win = no2_contrib_win + others_win
                # 목표: no2_contrib_new + others_win ≤ measured_win
                # no2_contrib_new = measured_win - others_win
                others_win = fitted_win - no2_cs_win * conc[no2_idx]
                no2_budget = measured_win - others_win
                if no2_budget > 0 and no2_cs_win * conc[no2_idx] > 0:
                    scale = no2_budget / (no2_cs_win * conc[no2_idx])
                    scale_factors.append(min(scale, 1.0))

        no2_original = conc[no2_idx]

        if not scale_factors:
            info = {
                'applied': False,
                'no2_original': no2_original,
                'scale_factor': 1.0,
                'signal_level': signal_level,
                'n_violated_windows': 0,
                'description': f'NO2 vis constraint: 위반 없음 (tolerance={self.no2_vis_tolerance:.0%})'
            }
            return conc_result, info

        # ── Robust 집계: 5th percentile ────────────────────────────────────────
        applied_scale = float(np.percentile(scale_factors, self.no2_vis_percentile))
        applied_scale = max(applied_scale, 0.0)  # 음수 방지

        if applied_scale < 1.0:
            no2_corrected = no2_original * applied_scale
            conc_result[no2_idx] = no2_corrected
            reduction_pct = (1 - applied_scale) * 100
            info = {
                'applied': True,
                'no2_original': no2_original,
                'no2_corrected': no2_corrected,
                'scale_factor': applied_scale,
                'reduction_pct': reduction_pct,
                'signal_level': signal_level,
                'n_violated_windows': len(scale_factors),
                'scale_min': min(scale_factors),
                'scale_median': float(np.median(scale_factors)),
                'window_nm': self.no2_vis_window_nm,
                'region': self.no2_vis_range,
                'description': (f'NO2 vis constraint: {no2_original:.2e} → {no2_corrected:.2e} '
                                f'(scale={applied_scale:.3f}, {reduction_pct:.1f}% 감소, '
                                f'{len(scale_factors)} 윈도우 위반, '
                                f'5th-pct 집계) [v8.18]')
            }
        else:
            info = {
                'applied': False,
                'no2_original': no2_original,
                'scale_factor': applied_scale,
                'signal_level': signal_level,
                'n_violated_windows': len(scale_factors),
                'description': f'NO2 vis constraint: 이미 만족 (5th-pct scale={applied_scale:.3f})'
            }

        return conc_result, info

    def correct_n2o4_by_equilibrium(self, conc: np.ndarray, fitting_info: dict) -> Tuple[np.ndarray, dict]:
        """
        NO2-N2O4 평형 관계를 이용한 N2O4 농도 상한 제한
        
        NNLS 피팅 후 N2O4 농도가 평형값보다 높으면,
        평형 농도를 상한으로 제한 (NO2는 변경하지 않음)
        
        Parameters:
        -----------
        conc : np.ndarray
            NNLS 피팅된 농도 배열 (molecules/cm³)
        fitting_info : dict
            피팅 정보 딕셔너리
            
        Returns:
        --------
        conc_corrected : np.ndarray
            보정된 농도 배열
        fitting_info : dict
            업데이트된 피팅 정보
        """
        idx_map = {sp: self.species_list.index(sp) for sp in self.species_list}
        no2_idx = idx_map.get('NO2')
        n2o4_idx = idx_map.get('N2O4')
        
        if no2_idx is None or n2o4_idx is None:
            return conc, fitting_info
        
        no2_conc = conc[no2_idx]
        n2o4_conc_fitted = conc[n2o4_idx]
        
        # NO2가 없거나 N2O4가 없으면 보정 불필요
        if no2_conc <= 0 or n2o4_conc_fitted <= 0:
            fitting_info['n2o4_equilibrium'] = {
                'correction_applied': False,
                'description': 'No correction needed (NO2 or N2O4 is zero)'
            }
            return conc, fitting_info
        
        # 평형 상수 계산
        Keq = self.calculate_n2o4_equilibrium_constant(self.temperature)
        
        # 평형 N2O4 농도 (상한) 계산: [N2O4]_max = Keq * [NO2]^2
        n2o4_max = Keq * (no2_conc ** 2)
        
        conc_corrected = conc.copy()
        
        # 피팅된 N2O4가 평형 상한을 초과하면 상한으로 제한
        if n2o4_conc_fitted > n2o4_max:
            conc_corrected[n2o4_idx] = n2o4_max
            
            fitting_info['n2o4_equilibrium'] = {
                'correction_applied': True,
                'temperature_K': self.temperature,
                'Keq': Keq,
                'n2o4_fitted': n2o4_conc_fitted,
                'n2o4_max': n2o4_max,
                'n2o4_corrected': n2o4_max,
                'ratio_before': n2o4_conc_fitted / n2o4_max,
                'no2_concentration': no2_conc,
                'description': f'N2O4 capped: {n2o4_conc_fitted:.2e} → {n2o4_max:.2e} (was {n2o4_conc_fitted/n2o4_max:.1f}x over limit)'
            }
        else:
            fitting_info['n2o4_equilibrium'] = {
                'correction_applied': False,
                'temperature_K': self.temperature,
                'Keq': Keq,
                'n2o4_fitted': n2o4_conc_fitted,
                'n2o4_max': n2o4_max,
                'ratio': n2o4_conc_fitted / n2o4_max if n2o4_max > 0 else 0,
                'description': f'N2O4 within limit ({n2o4_conc_fitted/n2o4_max:.1%} of max)'
            }
        
        return conc_corrected, fitting_info

    def apply_detection_limits(self, conc: np.ndarray, fitting_info: dict) -> Tuple[np.ndarray, dict]:
        """
        Detection limit 이하의 농도를 0으로 처리
        
        일반 UV-Vis 분광법 기준:
        - Path length: 5 cm
        - ΔA_min: 1×10⁻³ (typical detection limit)
        - DL = ΔA_min / (σ × L)
        
        Parameters:
        -----------
        conc : np.ndarray
            피팅된 농도 배열 (molecules/cm³)
        fitting_info : dict
            피팅 정보 딕셔너리
            
        Returns:
        --------
        conc_corrected : np.ndarray
            Detection limit 적용 후 농도 배열
        fitting_info : dict
            업데이트된 피팅 정보
        """
        conc_corrected = conc.copy()
        below_dl = {}
        
        for i, species in enumerate(self.species_list):
            if species in self.detection_limits:
                dl = self.detection_limits[species]
                if conc[i] < dl and conc[i] > 0:
                    below_dl[species] = {
                        'original': conc[i],
                        'detection_limit': dl,
                        'ratio': conc[i] / dl
                    }
                    conc_corrected[i] = 0
        
        if below_dl:
            fitting_info['detection_limit'] = {
                'applied': True,
                'species_below_dl': below_dl,
                'concentrations': conc_corrected.copy(),
                'r2': fitting_info.get('step4_equilibrium', fitting_info.get('step3_red', {})).get('r2', 0),
                'description': f'{len(below_dl)} species below detection limit → set to 0'
            }
        else:
            fitting_info['detection_limit'] = {
                'applied': False,
                'concentrations': conc_corrected.copy(),
                'r2': fitting_info.get('step4_equilibrium', fitting_info.get('step3_red', {})).get('r2', 0),
                'description': 'All species above detection limit'
            }
        
        return conc_corrected, fitting_info

    def get_analysis_mode(self) -> str:
        """
        분석 모드 선택 (단일/배치)
        """
        print("\n" + "="*60)
        print("분석 모드 선택")
        print("="*60)
        print("\n1. 단일 파일 분석 (상세 피팅 그래프)")
        print("2. 배치 분석 (폴더 내 모든 파일)")

        while True:
            choice = input("\n선택 (1 또는 2): ").strip()
            if choice == '1':
                return 'single'
            elif choice == '2':
                return 'batch'
            else:
                print("⚠ 1 또는 2를 입력해주세요.")

    def get_single_file_path(self, folder_path: str) -> Optional[str]:
        """
        단일 파일 경로 입력 받기
        """
        print("\n" + "="*60)
        print("파일 선택")
        print("="*60)

        # === MODIFIED: 후보 리스트 필터 (Absorbance 포함 + __<정수>__) ===
        txt_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith('.txt')
                     and ('absorbance' in f.lower())
                     and self.TIME_TOKEN.search(f)]

        if txt_files:
            print(f"\n폴더 내 선택 가능한 Absorbance 파일 ({len(txt_files)}개):")
            for i, f in enumerate(sorted(txt_files)[:10], 1):  # 처음 10개만 표시
                print(f"  {i}. {f}")
            if len(txt_files) > 10:
                print(f"  ... 외 {len(txt_files)-10}개")

        print("\n분석할 파일명을 입력하세요.")
        print("(전체 경로 또는 파일명만 입력)")

        while True:
            file_input = input("\n파일명: ").strip().strip('"').strip("'")

            # 전체 경로로 입력한 경우
            if os.path.exists(file_input):
                return file_input

            # 파일명만 입력한 경우
            file_path = os.path.join(folder_path, file_input)
            if os.path.exists(file_path):
                return file_path

            print(f"⚠ 파일을 찾을 수 없습니다: {file_input}")

    def get_batch_plot_option(self) -> bool:
        """
        배치 분석 시 개별 파일 플롯 생성 여부 선택
        """
        print("\n" + "="*60)
        print("개별 파일 플롯 옵션")
        print("="*60)
        print("\n배치 분석 시 각 Absorbance 파일별 분석 플롯을 생성하시겠습니까?")
        print("  1. 예 - 각 파일마다 PNG 플롯 생성 (시간 소요)")
        print("  2. 아니오 - 최종 요약 플롯만 생성 (빠름)")
        
        while True:
            choice = input("\n선택 (1 또는 2): ").strip()
            if choice == '1':
                print("✓ 각 파일별 분석 플롯을 생성합니다.")
                return True
            elif choice == '2':
                print("✓ 최종 요약 플롯만 생성합니다.")
                return False
            else:
                print("⚠ 1 또는 2를 입력해주세요.")

    def get_humidity_weight(self):
        """
        사용자로부터 상대습도(RH, %) 입력 받기
        
        [v8.11] RH(%) 직접 입력으로 변경
        - humidity_weight = RH / 100  (내부 변환)
        - HONO 상한: N2O4 농도 × humidity_weight  (N2O4+H2O→HONO+HNO3, 1:1)
        - HONO2 상한: N2O5 농도 × 2 × humidity_weight  (N2O5+H2O→2HNO3, 1:2)
        - 근거: Stutz et al. (2004) — [HONO]/[NO2] ∝ RH (선형)
                Finlayson-Pitts et al. (2003) — N2O4가 HONO의 핵심 전구체
        """
        print("\n" + "="*60)
        print("습도 설정 [v8.11]")
        print("="*60)
        print("\n모든 파일에 동일한 습도 설정을 적용합니다.")
        print("HONO와 HONO2는 수분과의 반응으로 생성되는 2차 생성물입니다.")
        print("\n상대습도(RH, %)를 입력하세요:")
        print("  -   0%: 완전 건조 (HONO/HONO2 제외)")
        print("  -  30%: 약한 습도")
        print("  -  50%: 중간 습도")
        print("  -  70%: 높은 습도")
        print("  - 100%: 포화 수증기")
        print("\n[HONO/HONO2 농도 상한 적용 방식]")
        print("  HONO  상한 = [N2O4] × (RH/100)   (N2O4+H2O → HONO+HNO3)")
        print("  HONO2 상한 = [N2O5] × 2 × (RH/100) (N2O5+H2O → 2HNO3)")

        while True:
            try:
                rh_input = input("\n상대습도 RH (0-100 %): ")
                rh_value = float(rh_input)

                if 0.0 <= rh_value <= 100.0:
                    self.humidity_weight = rh_value / 100.0   # 내부 변환

                    if rh_value == 0.0:
                        print(f"✓ 완전 건조 (HONO/HONO2 제외)")
                    elif rh_value < 20.0:
                        print(f"✓ 매우 건조한 조건 (RH={rh_value:.0f}%, weight={self.humidity_weight:.2f})")
                    elif rh_value < 40.0:
                        print(f"✓ 건조한 조건 (RH={rh_value:.0f}%, weight={self.humidity_weight:.2f})")
                    elif rh_value < 60.0:
                        print(f"✓ 보통 습도 (RH={rh_value:.0f}%, weight={self.humidity_weight:.2f})")
                    elif rh_value < 80.0:
                        print(f"✓ 습한 조건 (RH={rh_value:.0f}%, weight={self.humidity_weight:.2f})")
                    else:
                        print(f"✓ 매우 습한 조건 (RH={rh_value:.0f}%, weight={self.humidity_weight:.2f})")

                    print(f"\n[v8.11] HONO/HONO2 피팅 방식:")
                    print(f"  Step1 UV: HONO, HONO2를 O3/NO/N2O5와 함께 피팅 (cross section 원본)")
                    print(f"  Step2 Vis: HONO 350~396nm 잔차 보완 피팅 (cross section 원본)")
                    print(f"  전구체 상한: HONO ≤ [N2O4]×{self.humidity_weight:.2f}, "
                          f"HONO2 ≤ [N2O5]×2×{self.humidity_weight:.2f}")
                    break
                else:
                    print("⚠ 0부터 100 사이의 값을 입력해주세요.")
            except ValueError:
                print("⚠ 올바른 숫자를 입력해주세요. (예: 50)")
            except KeyboardInterrupt:
                print("\n\n기본값 50% (중간 습도)으로 설정합니다.")
                self.humidity_weight = 0.5
                break

    def apply_hono_precursor_cap(self, conc: np.ndarray, fitting_info: dict) -> Tuple[np.ndarray, dict]:
        """
        [v8.14] HONO/HONO2 전구체 기반 농도 상한 적용 (전구체 교체)

        [v8.11 문제]
          HONO_max = [N2O4] × humidity_weight
          → N2O4 평형 cap(Step4) 후 N2O4 ≈ 0 → HONO_max ≈ 0 → HONO 연쇄 소거
          → 실온(295K)에서 N2O4_eq가 DL 이하이므로 HONO 검출 불가능한 구조

        [v8.14 수정]
          HONO_max  = [NO2] × hono_no2_factor × (RH/100)
            근거: 2NO2 + H2O → HONO + HNO3  (주 반응 경로, Finlayson-Pitts 2003)
                  [HONO]/[NO2] ∝ RH  실험적 선형 관계 (Stutz et al. 2004)
                  hono_no2_factor k = 0.003~0.05 (환경·표면 의존, 기본 0.01)
          HONO2_max = [N2O5] × 2 × (RH/100)   (변경 없음)
            근거: N2O5 + H2O → 2 HNO3  (1:2 화학량론)

        [v8.14 순서 수정]
          호출 위치: Step4 평형 보정(N2O4/N2O5 cap) 이후로 이동
          → cap된 NO2/N2O5 기준으로 상한 계산 (불일치 해소)

        참고:
          Finlayson-Pitts et al. (2003) Phys. Chem. Chem. Phys. 5, 223-242
          Stutz et al. (2004) J. Geophys. Res. 109, D03307
        """
        idx_map  = {sp: self.species_list.index(sp) for sp in self.species_list}
        hono_idx  = idx_map.get('HONO')
        hono2_idx = idx_map.get('HONO2')
        no2_idx   = idx_map.get('NO2')
        n2o5_idx  = idx_map.get('N2O5')

        conc_corrected = conc.copy()
        cap_info = {}

        # ── HONO 상한: [NO2] × hono_no2_factor × RH  [v8.14 변경] ─────
        if hono_idx is not None and no2_idx is not None:
            hono_fitted = conc[hono_idx]
            hono_max    = conc[no2_idx] * self.hono_no2_factor * self.humidity_weight
            if hono_fitted > hono_max and hono_max >= 0:
                conc_corrected[hono_idx] = hono_max
                cap_info['HONO'] = {
                    'applied': True,
                    'fitted': hono_fitted,
                    'cap': hono_max,
                    'no2': conc[no2_idx],
                    'hono_no2_factor': self.hono_no2_factor,
                    'humidity_weight': self.humidity_weight,
                    'description': (f'HONO capped: {hono_fitted:.2e} → {hono_max:.2e} '
                                    f'([NO2]×{self.hono_no2_factor}×{self.humidity_weight:.2f}) [v8.14]')
                }
            else:
                cap_info['HONO'] = {
                    'applied': False,
                    'fitted': hono_fitted,
                    'cap': hono_max,
                    'no2': conc[no2_idx]
                }

        # ── HONO2 상한: [N2O5] × 2 × RH  (변경 없음) ────────────────
        if hono2_idx is not None and n2o5_idx is not None:
            hono2_fitted = conc[hono2_idx]
            hono2_max    = conc[n2o5_idx] * 2.0 * self.humidity_weight
            if hono2_fitted > hono2_max and hono2_max >= 0:
                conc_corrected[hono2_idx] = hono2_max
                cap_info['HONO2'] = {
                    'applied': True,
                    'fitted': hono2_fitted,
                    'cap': hono2_max,
                    'n2o5': conc[n2o5_idx],
                    'humidity_weight': self.humidity_weight,
                    'description': (f'HONO2 capped: {hono2_fitted:.2e} → {hono2_max:.2e} '
                                    f'([N2O5]×2×{self.humidity_weight:.2f})')
                }
            else:
                cap_info['HONO2'] = {
                    'applied': False,
                    'fitted': hono2_fitted,
                    'cap': hono2_max
                }

        fitting_info['hono_precursor_cap'] = cap_info

        for sp, info in cap_info.items():
            if info.get('applied'):
                print(f"  [전구체 상한] {info['description']}")

        return conc_corrected, fitting_info

    def get_folder_path(self) -> str:
        """
        사용자로부터 폴더 경로 입력 받기
        """
        print("\n" + "="*60)
        print("폴더 경로 입력")
        print("="*60)
        print("\n분석할 Absorbance 파일들이 있는 폴더 경로를 입력하세요.")
        print("예시: C:\\Users\\asap\\Desktop\\NNLS_A.I\\measured\\10000_1")

        while True:
            folder_path = input("\n폴더 경로: ").strip()

            # 따옴표 제거
            folder_path = folder_path.strip('"').strip("'")

            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # === MODIFIED: 유효 파일 판정 (Absorbance + __<정수>__) ===
                txt_files = [f for f in os.listdir(folder_path)
                             if f.lower().endswith('.txt')
                             and ('absorbance' in f.lower())
                             and self.TIME_TOKEN.search(f)]
                if txt_files:
                    print(f"\n✓ {len(txt_files)}개 Absorbance 파일 발견 (패턴: 'Absorbance' + '__<정수>__')")
                    return folder_path
                else:
                    print("⚠ 해당 폴더에 유효한 Absorbance 파일이 없습니다.")
                    print("   (조건: 파일명에 'Absorbance' 포함 + '__<정수>__' 토큰 포함)")
            else:
                print("⚠ 올바른 폴더 경로가 아닙니다.")

    # === MODIFIED: 파일명에서 시간 추출 규칙 완화 ===
    def extract_time_from_filename(self, filename: str) -> Optional[float]:
        """
        파일명에서 시간 추출
        - 규칙: 파일명 어디든 '__<정수>__' 패턴을 찾음
        - 여러 개가 있으면 마지막 토큰 사용
        예시:
          Absorbance-baseline__0__.txt -> 0
          Absorbance-water__3__.txt -> 3
          sample__Absorbance__15__.txt -> 15
        """
        matches = self.TIME_TOKEN.findall(filename)
        if matches:
            return float(matches[-1])
        return None

    def load_cross_sections(self):
        """
        Cross section 데이터 로드
        """
        if self.cross_sections:
            return

        print("\n" + "="*60)
        print("Cross Section 데이터 로딩 (8종)")
        print("="*60)

        files = [f for f in os.listdir(self.cross_section_path)
                 if f.endswith('_ordered_cross_section.txt')]

        for species in self.species_order:
            for file in files:
                if file.startswith(species + '_') or file.startswith(species + '-'):
                    filepath = os.path.join(self.cross_section_path, file)
                    data = self.load_data_file(filepath)
                    if data is not None:
                        self.cross_sections[species] = data
                        self.species_list.append(species)
                        print(f"  ✓ {species}: {len(data)} points")
                    break

        print(f"\n로드 완료: {len(self.species_list)}개 화학종")

    def load_data_file(self, filepath: str) -> pd.DataFrame:
        """
        데이터 파일 로드
        - Absorbance 파일: 헤더가 있고 '>>>>>Begin Spectral Data<<<<<' 이후부터 사용
        - Cross section 파일: 헤더 없이 바로 두 컬럼(wavelength, value)
        둘 다 처리 가능하도록 구현
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 기본값: 파일 처음부터 읽기
            start_idx = 0

            # Absorbance 파일처럼 헤더가 있는 경우: Spectral Data 시작 위치 찾기
            for i, line in enumerate(lines):
                if "Begin Spectral Data" in line:
                    start_idx = i + 1
                    break

            wavelengths = []
            values = []

            for line in lines[start_idx:]:
                parts = line.strip().split()
                # 숫자 두 개만 있는 줄만 사용
                if len(parts) != 2:
                    continue
                try:
                    wl = float(parts[0])
                    val = float(parts[1])
                    wavelengths.append(wl)
                    values.append(val)
                except:
                    continue

            if wavelengths:
                return pd.DataFrame({
                    'wavelength': wavelengths,
                    'value': values
                }).sort_values('wavelength').reset_index(drop=True)
            else:
                print(f"[경고] 유효한 데이터가 없습니다: {filepath}")
                return None
        except Exception as e:
            print(f"[에러] 파일 로드 실패: {filepath}\n{e}")
            return None

    def detect_plasma_type(self, wavelengths: np.ndarray, absorbance: np.ndarray) -> Dict:
        """
        플라즈마 타입 판별
        """
        ozone_range = (wavelengths >= 250) & (wavelengths <= 260)
        ozone_peak = {'has_peak': False, 'intensity': 0, 'is_dominant': False, 'ratio': 0}

        if np.any(ozone_range):
            ozone_abs = absorbance[ozone_range]
            ozone_wl = wavelengths[ozone_range]

            max_idx = np.argmax(ozone_abs)
            peak_wl = ozone_wl[max_idx]
            peak_intensity = ozone_abs[max_idx]

            total_max = np.max(absorbance) if len(absorbance) > 0 else 0
            ozone_ratio = peak_intensity / total_max if total_max > 0 else 0

            ozone_peak = {
                'has_peak': peak_intensity > 0.01,
                'wavelength': peak_wl,
                'intensity': peak_intensity,
                'ratio': ozone_ratio,
                'is_dominant': ozone_ratio > 0.5
            }

        no2_range = (wavelengths >= 370) & (wavelengths <= 390)
        no2_intensity = np.mean(absorbance[no2_range]) if np.any(no2_range) else 0

        if ozone_peak.get('is_dominant', False):
            plasma_type = 'O3_dominant'
        elif ozone_peak.get('ratio', 0) < 0.2 and no2_intensity > 0.01:
            plasma_type = 'NOx_dominant'
        else:
            plasma_type = 'mixed'

        return {
            'type': plasma_type,
            'ozone_peak': ozone_peak,
            'no2_intensity': no2_intensity
        }


    def improved_nnls_fitting(self, A: np.ndarray, b: np.ndarray,
                              wavelengths: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        [v7] 2단계 피팅 알고리즘 (확장)
        1단계: UV(210-350nm) → O3, NO, N2O5
        2단계: Visible(350-650nm) → NO2, N2O4, NO3 (+HONO)
        """
        fitting_info = {}
        conc = np.zeros(len(self.species_list))
        
        # 인덱스 가져오기
        idx_map = {sp: self.species_list.index(sp) for sp in self.species_list}
        o3_idx = idx_map.get('O3')
        no_idx = idx_map.get('NO')
        no2_idx = idx_map.get('NO2')
        no3_idx = idx_map.get('NO3')
        n2o4_idx = idx_map.get('N2O4')
        n2o5_idx = idx_map.get('N2O5')
        hono_idx = idx_map.get('HONO')
        hono2_idx = idx_map.get('HONO2')

        # ===== 1단계: UV 영역 (uv_start-350nm) =====
        # [v8.13] N2O4를 Step1에 추가
        #   근거: N2O4 cross section이 200-230nm에서 O3의 6.5배 (5.4e-17 vs 8.4e-18 cm²)
        #         Step1에서 N2O4를 제외하면 해당 흡수가 O3로 잘못 귀속되어
        #         200-230nm underfit 발생 → N2O4를 UV에서 직접 정량
        #   N2O4 흡수 범위: 185-390nm (Step1 전체 구간에 걸침)
        #
        # [v8.14] Step1 UV에서 HONO/HONO2 제외 (humid 조건 포함)
        #
        # [v8.11 방식의 문제]
        #   HONO CS와 N2O5 CS가 UV(200-250nm)에서 상관계수 0.91로 매우 유사
        #   → humid Step1에서 NNLS가 N2O5 대신 HONO를 선택 (267ppm 과추정)
        #   → precursor cap으로 HONO=0이 되면 200-230nm에 실제 underfit 발생
        #   → N2O5도 0으로 소거되는 연쇄 문제
        #
        # [v8.14 수정]
        #   HONO/HONO2는 Step1(UV)에서 완전 제외 (humid 무관)
        #   HONO는 Step2 Vis(350-396nm) 잔차에서만 결정
        #   물리적 근거:
        #     N2O5: 기상 반응(NO2+NO3 → N2O5) → UV에서 직접 정량 필요
        #     HONO: 표면 반응(2NO2+H2O → HONO) → UV에서 N2O5와 구분 불가
        #           Vis(350-396nm) vibronic band에서 더 신뢰성 있게 결정
        uv_region = (wavelengths >= self.uv_start) & (wavelengths <= 350)

        # Step1 습도 종 목록: 항상 빈 리스트 (HONO/HONO2는 Step2에서만 결정)
        humid_uv_species = []

        if np.any(uv_region):
            A_uv = A[uv_region]
            b_uv = b[uv_region]

            if self.n2o5_scale == 0.0:
                # N2O5 제외: O3, NO, N2O4 [+HONO, HONO2] 피팅
                # [v8.13] N2O4 추가
                base_sp = ['O3', 'NO', 'N2O4'] + humid_uv_species
                stage1_indices_no_n2o5 = [idx_map[sp] for sp in base_sp if sp in self.species_list]

                A_stage1 = A_uv[:, stage1_indices_no_n2o5]
                conc_stage1, _ = nnls(A_stage1, b_uv)

                for i, idx in enumerate(stage1_indices_no_n2o5):
                    conc[idx] = conc_stage1[i]
                if n2o5_idx is not None:
                    conc[n2o5_idx] = 0

            elif self.n2o5_scale < 1.0:
                # N2O5 스케일 제한: 먼저 전체 피팅 후 N2O5 스케일 다운
                # [v8.13] N2O4 추가
                base_sp = ['O3', 'NO', 'N2O4', 'N2O5'] + humid_uv_species
                stage1_indices = [idx_map[sp] for sp in base_sp if sp in self.species_list]

                A_stage1 = A_uv[:, stage1_indices]
                conc_stage1_full, _ = nnls(A_stage1, b_uv)

                # N2O5 스케일 적용
                n2o5_local_idx = [i for i, idx in enumerate(stage1_indices) if idx == n2o5_idx][0]
                n2o5_scaled = conc_stage1_full[n2o5_local_idx] * self.n2o5_scale

                # N2O5 고정 후 나머지(O3, NO, N2O4 [+HONO, HONO2]) 재피팅
                n2o5_contribution = A_uv[:, n2o5_idx] * n2o5_scaled
                residual = b_uv - n2o5_contribution
                residual = np.maximum(residual, 0)

                stage1_indices_no_n2o5 = [idx for idx in stage1_indices if idx != n2o5_idx]
                A_stage1_no_n2o5 = A_uv[:, stage1_indices_no_n2o5]
                conc_o3_no, _ = nnls(A_stage1_no_n2o5, residual)

                for i, idx in enumerate(stage1_indices_no_n2o5):
                    conc[idx] = conc_o3_no[i]
                conc[n2o5_idx] = n2o5_scaled

            else:
                # 기본 방식: O3, NO, N2O4, N2O5 [+HONO, HONO2] 모두 피팅
                # [v8.13] N2O4 추가
                base_sp = ['O3', 'NO', 'N2O4', 'N2O5'] + humid_uv_species
                stage1_indices = [idx_map[sp] for sp in base_sp if sp in self.species_list]

                A_stage1 = A_uv[:, stage1_indices]
                conc_stage1, _ = nnls(A_stage1, b_uv)

                for i, idx in enumerate(stage1_indices):
                    conc[idx] = conc_stage1[i]

        # stage1_indices 재정의 (O3-NO 규칙 및 이후 코드에서 사용)
        # [v8.13] N2O4 포함
        base_sp_for_stage1 = ['O3', 'NO', 'N2O4', 'N2O5'] + humid_uv_species
        stage1_indices = [idx_map[sp] for sp in base_sp_for_stage1 if sp in self.species_list]

        fitted_stage1 = A @ conc
        r2_stage1 = 1 - np.sum((fitted_stage1 - b)**2) / np.sum((b - np.mean(b))**2)

        hono_in_step1 = 'HONO' in humid_uv_species
        fitting_info['step1'] = {
            'concentrations': conc.copy(),
            'r2': r2_stage1,
            'description': (f'Stage1: UV fitting (N2O5 scale={self.n2o5_scale}'
                            f'{", +HONO/HONO2" if hono_in_step1 else ""})')
        }

        # ===== O3-NO 경쟁 규칙 적용 =====
        # [v8.14] v8.7 방식(절대 농도 기준)으로 복원
        #
        # 반응: NO + O3 → NO2 + O2  (k = 1.8e-14 × exp(-1500/T) cm³/molecule/s, Atkinson 2004)
        # 물리적 근거:
        #   T=298K, [O3]=2e14: k≈2.0e-14 → τ = 1/(k×[O3]) ≈ 0.25 s (수초 내 NO 소멸)
        #   → O3 >= 2e14 (~8 ppm) 이면 NO는 정상 상태에서 검출 불가
        #
        # v8.12 몰비 기준([O3]/[NO] >= 10)의 문제:
        #   NO cross section이 200-235nm에만 존재하므로, n2o5_scale < 1.0 설정 시
        #   200-230nm 잔차를 NNLS가 NO로 귀속 → NO 과추정 → 비율이 항상 5~6에 머뭄
        #   → O3가 2500 ppm이어도 규칙이 발동되지 않는 구조적 문제
        #
        # v8.7 기준: [O3] >= 2e14 (~8 ppm) 이면 무조건 NO = 0
        #   → NO 과추정 여부와 무관하게 물리적으로 타당한 결과 보장
        #
        # 참고: Atkinson et al., Atmos. Chem. Phys. 4 (2004) 1461-1738
        O3_ABS_THRESHOLD = 2e14   # [v8.14] v8.7 방식: O3 절대 농도 기준 (≈ 8 ppm)
        o3_no_applied = False

        if o3_idx is not None and no_idx is not None:
            o3_conc = conc[o3_idx]
            no_conc = conc[no_idx]
            if o3_conc >= O3_ABS_THRESHOLD and no_conc > 0:
                o3_no_applied = True

                if self.n2o5_scale < 1.0:
                    # N2O5 기여도 고정, O3만 재피팅 (NO 제거)
                    # [v8.11] HONO/HONO2는 Step1에서 이미 결정된 값 유지
                    n2o5_current = conc[n2o5_idx] if n2o5_idx is not None else 0
                    n2o5_contribution = (A[uv_region][:, n2o5_idx] * n2o5_current
                                         if n2o5_idx is not None else 0)

                    # HONO/HONO2 기여도도 고정
                    hono_contribution = np.zeros(np.sum(uv_region))
                    for sp in humid_uv_species:
                        sp_idx = idx_map.get(sp)
                        if sp_idx is not None:
                            hono_contribution += A[uv_region][:, sp_idx] * conc[sp_idx]

                    residual_for_o3 = b[uv_region] - n2o5_contribution - hono_contribution
                    residual_for_o3 = np.maximum(residual_for_o3, 0)

                    A_o3_only = A[uv_region][:, [o3_idx]]
                    conc_o3_only, _ = nnls(A_o3_only, residual_for_o3)

                    conc[o3_idx] = conc_o3_only[0]
                    conc[no_idx] = 0
                    # N2O5, HONO, HONO2는 이미 설정된 값 유지
                else:
                    # O3, N2O5 [+HONO, HONO2] 함께 재피팅 (NO 제외)
                    stage1_no_NO = [idx for idx in stage1_indices if idx != no_idx]

                    if len(stage1_no_NO) > 0:
                        A_stage1_no_NO = A[uv_region][:, stage1_no_NO]
                        conc_stage1_no_NO, _ = nnls(A_stage1_no_NO, b[uv_region])

                        conc[no_idx] = 0
                        for i, idx in enumerate(stage1_no_NO):
                            conc[idx] = conc_stage1_no_NO[i]

                        # [v8.17] n2o5_scale 재적용 버그 수정
                        # O3-NO 규칙 발동 시 자유 재피팅으로 n2o5_scale이 무시되던 문제 수정.
                        # n2o5_scale < 1.0이면: 재피팅된 N2O5를 스케일다운 후
                        # 나머지 종(O3, N2O4 등)을 잔차로 재피팅하여 scale을 강제 적용.
                        if n2o5_idx is not None and self.n2o5_scale < 1.0:
                            n2o5_after_rule = conc[n2o5_idx] * self.n2o5_scale
                            n2o5_contrib = A[uv_region][:, n2o5_idx] * n2o5_after_rule
                            residual2 = b[uv_region] - n2o5_contrib
                            residual2 = np.maximum(residual2, 0)
                            refit_ids = [idx for idx in stage1_no_NO if idx != n2o5_idx]
                            if refit_ids:
                                conc_refit, _ = nnls(A[uv_region][:, refit_ids], residual2)
                                for i, idx in enumerate(refit_ids):
                                    conc[idx] = conc_refit[i]
                            conc[n2o5_idx] = n2o5_after_rule
        
        fitting_info['step1b'] = {
            'concentrations': conc.copy(),
            'r2': 1 - np.sum((A @ conc - b)**2) / np.sum((b - np.mean(b))**2),
            'description': f'O3-NO rule {"applied" if o3_no_applied else "not needed"}',
            'o3_no_applied': o3_no_applied
        }

        # ===== 2단계: Visible 영역 (350-650nm) - NO2, NO3 (+HONO 잔차 보완) =====
        # [v8.13] N2O4를 Step1(UV)에서 정량하므로 Step2에서 제외
        #   - N2O4는 Step1 잔차에 이미 반영됨 → Step2 잔차에서 중복 피팅 방지
        #   - Step2 종: NO2, NO3 [+HONO if humid]
        #   - NO3는 580-640nm 피크가 주 정량 구간이므로 Step2에 유지
        # [v8.11] HONO cross section 원본 그대로 사용 (× 0.3 제거)
        #   - HONO의 350-396nm 흡수(~20%)를 Step1 잔차에서 보완
        #   - HONO2는 350nm 이상 흡수 없으므로 Step2에서 제외
        #   - 피팅 후 apply_hono_precursor_cap()으로 전구체 상한 적용
        vis_region = (wavelengths >= 350) & (wavelengths <= 650)

        if np.any(vis_region):
            fitted_so_far = A @ conc
            residual_vis = b[vis_region] - fitted_so_far[vis_region]
            residual_vis[residual_vis < 0] = 0

            if self.humidity_weight > 0:
                # 습도 있음: NO2 + NO3 + HONO (cross section 원본)
                # [v8.13] N2O4 제외 (Step1에서 이미 정량)
                stage2_indices = []
                A_stage2_list = []

                if no2_idx is not None:
                    stage2_indices.append(no2_idx)
                    A_stage2_list.append(A[vis_region, no2_idx])

                if no3_idx is not None:
                    stage2_indices.append(no3_idx)
                    A_stage2_list.append(A[vis_region, no3_idx])

                # HONO: 350-396nm 잔여 흡수 보완 (원본 cross section)
                if hono_idx is not None:
                    stage2_indices.append(hono_idx)
                    A_stage2_list.append(A[vis_region, hono_idx])

                # HONO2: 350nm 이상 흡수 없으므로 Step2에서 제외
                # (Step1에서 이미 결정된 값 유지)

                if len(A_stage2_list) > 0:
                    A_stage2 = np.column_stack(A_stage2_list)
                    conc_stage2, _ = nnls(A_stage2, residual_vis)

                    for i, idx in enumerate(stage2_indices):
                        conc[idx] = conc_stage2[i]
            else:
                # 습도 0: NO2 + NO3만
                # [v8.13] N2O4 제외 (Step1에서 이미 정량)
                stage2_indices = []
                A_stage2_list = []

                if no2_idx is not None:
                    stage2_indices.append(no2_idx)
                    A_stage2_list.append(A[vis_region, no2_idx])

                if no3_idx is not None:
                    stage2_indices.append(no3_idx)
                    A_stage2_list.append(A[vis_region, no3_idx])

                if len(A_stage2_list) > 0:
                    A_stage2 = np.column_stack(A_stage2_list)
                    conc_stage2, _ = nnls(A_stage2, residual_vis)

                    for i, idx in enumerate(stage2_indices):
                        conc[idx] = conc_stage2[i]

                if hono_idx is not None:
                    conc[hono_idx] = 0
                if hono2_idx is not None:
                    conc[hono2_idx] = 0

        fitted_stage2 = A @ conc
        r2_stage2 = 1 - np.sum((fitted_stage2 - b)**2) / np.sum((b - np.mean(b))**2)

        fitting_info['step2'] = {
            'concentrations': conc.copy(),
            'r2': r2_stage2,
            'description': (f'Stage2: Visible fitting (NO2, N2O4, NO3'
                            f'{", +HONO(350-396nm residual)" if self.humidity_weight > 0 else ""})')
        }

        # ── [v8.14] NO2 Vis Overfit 제약 (Step2 완료 후 즉시) ────────────
        # 350-450nm에서 NO2 vibronic 구조 + 노이즈로 인한 fitted>measured 보정
        if no2_idx is not None and conc[no2_idx] > 0:
            conc, no2_vis_info = self.apply_no2_vis_constraint(wavelengths, b, A, conc, no2_idx)
            fitting_info['step2_no2_vis'] = no2_vis_info
            if no2_vis_info.get('applied'):
                fitted_after_no2_vis = A @ conc
                r2_after_no2_vis = 1 - np.sum((fitted_after_no2_vis - b)**2) / np.sum((b - np.mean(b))**2)
                fitting_info['step2_no2_vis']['r2'] = r2_after_no2_vis

        # ── [v8.14] apply_hono_precursor_cap은 Step4 평형 보정 이후로 이동 ──
        # (N2O4/N2O5 cap된 값 기준으로 HONO/HONO2 상한 계산하기 위해)

        # ===== 3단계: Overfitting 보정 (습도>0일 때만) =====
        # [v8.11] HONO2는 370-400nm 흡수가 없으므로 스케일 보정 대상에서 제외
        #         HONO2는 전구체 상한(apply_hono_precursor_cap)으로만 제어
        if self.correct_overfit and self.humidity_weight > 0:
            no2_range = (wavelengths >= 370) & (wavelengths <= 400)

            if np.any(no2_range) and no2_idx is not None:
                fitted_no2_region = fitted_stage2[no2_range]
                measured_no2_region = b[no2_range]
                measured_mean = np.mean(measured_no2_region)

                if measured_mean >= 0.0005:
                    overfitting_ratio = np.mean(fitted_no2_region) / measured_mean

                    if overfitting_ratio > 1.3:
                        other_contrib = np.zeros(np.sum(no2_range))
                        for i, sp in enumerate(self.species_list):
                            if sp not in ['NO2', 'HONO']:   # HONO2 제외
                                other_contrib += A[no2_range, i] * conc[i]

                        target = np.mean(np.maximum(measured_no2_region - other_contrib, 0))

                        current_no2  = np.mean(A[no2_range, no2_idx] * conc[no2_idx])
                        current_hono = 0
                        if hono_idx is not None:
                            current_hono = np.mean(A[no2_range, hono_idx] * conc[hono_idx])

                        current_total = current_no2 + current_hono

                        if current_total > 0:
                            scale = min(target / current_total, 1.0)

                            conc[no2_idx] *= scale
                            if hono_idx is not None:
                                conc[hono_idx] *= scale
                            # [v8.11] HONO2는 여기서 스케일하지 않음
                            #         (전구체 상한에서 이미 제어됨)

                            fitted_stage3 = A @ conc
                            new_ratio = np.mean(fitted_stage3[no2_range]) / measured_mean

                            fitting_info['step3'] = {
                                'concentrations': conc.copy(),
                                'r2': 1 - np.sum((fitted_stage3 - b)**2) / np.sum((b - np.mean(b))**2),
                                'description': (f'Stage3: Overfit corrected '
                                                f'({overfitting_ratio:.1f}x -> {new_ratio:.1f}x)'
                                                f' [HONO2 excluded from scale]'),
                                'hono_adjusted': True
                            }

        # ===== 3단계 추가: 500-650nm under-fitting 보정 =====
        # [v8.14] NO2 제외: NO3, N2O4만으로 재피팅
        #
        # [v8.12 → v8.14 변경 이유]
        #   기존: NO2, N2O4, NO3를 함께 재피팅
        #   문제: NO2가 올라가면 350-450nm 전 구간에서 fitted>measured 발생
        #         NO2 Vis 제약(350-450nm)과 step3_red가 서로 역방향으로 작동
        #         → NO2 Vis 제약으로 낮춘 NO2를 step3_red가 다시 올림
        #
        # [v8.14 수정]
        #   NO2 제외 → NO3, N2O4만 재피팅
        #   근거: 500-650nm에서 NO3 CS가 강함 (2.9e-18 cm², peak@623nm)
        #         NO2 CS는 이 구간에서 꼬리(7.9e-20 cm²)에 불과
        #         N2O4는 400nm 이상 흡수 없으므로 사실상 NO3만 기여
        #   결과: 500-650nm underfit이 완전히 해소되지는 않지만
        #         350-450nm overfit(더 심각한 문제) 방지
        red_region = (wavelengths >= 500) & (wavelengths <= 650)

        if np.any(red_region) and no3_idx is not None:
            fitted_current = A @ conc
            meas_red = np.mean(b[red_region])
            fit_red  = np.mean(fitted_current[red_region])

            if meas_red > 0.001 and fit_red > 0:
                ratio_red = fit_red / meas_red
                if ratio_red < 0.95:  # under-fitting 감지
                    # [v8.14] 재피팅 대상: NO3, N2O4만 (NO2 제외)
                    # NO2를 포함하면 350-450nm 과추정 문제 발생
                    red_refit_species = [sp for sp in ['N2O4', 'NO3']
                                         if sp in self.species_list]
                    red_refit_indices = [idx_map[sp] for sp in red_refit_species]

                    # 재피팅 대상 제외한 나머지 기여 계산
                    other_conc_red = conc.copy()
                    for idx in red_refit_indices:
                        other_conc_red[idx] = 0
                    fitted_others_red = A[red_region] @ other_conc_red

                    # 잔차 (음수 클리핑)
                    residual_red = np.maximum(b[red_region] - fitted_others_red, 0)

                    # NNLS 재피팅
                    A_red_refit = A[red_region][:, red_refit_indices]
                    if A_red_refit.shape[1] > 0:
                        conc_red_new, _ = nnls(A_red_refit, residual_red)

                        # 농도 업데이트 (기존보다 감소는 허용하지 않음)
                        for i, idx in enumerate(red_refit_indices):
                            conc[idx] = max(conc[idx], conc_red_new[i])

                        fitted_after_red_correction = A @ conc
                        new_ratio_red = np.mean(fitted_after_red_correction[red_region]) / meas_red

                        fitting_info['step3_red'] = {
                            'concentrations': conc.copy(),
                            'r2': 1 - np.sum((fitted_after_red_correction - b)**2) / np.sum((b - np.mean(b))**2),
                            'description': (f'Stage3-Red: NNLS refit N2O4/NO3 (NO2 excluded) '
                                            f'({ratio_red:.2f}x -> {new_ratio_red:.2f}x) [v8.14]'),
                            'no3_adjusted': True,
                            'refit_species': red_refit_species
                        }

        # ===== N2O4 / N2O5 평형 보정 (Step 4) [v8.9: N2O5 추가] =====
        if self.correct_n2o4_equilibrium:
            conc, fitting_info = self.correct_n2o4_by_equilibrium(conc, fitting_info)
            conc, fitting_info = self.correct_n2o5_by_equilibrium(conc, fitting_info)  # [v8.9]

            n2o4_corrected = fitting_info.get('n2o4_equilibrium', {}).get('correction_applied', False)
            n2o5_corrected = fitting_info.get('n2o5_equilibrium', {}).get('correction_applied', False)

            if n2o4_corrected or n2o5_corrected:
                fitted_after_eq = A @ conc
                r2_after_eq = 1 - np.sum((fitted_after_eq - b)**2) / np.sum((b - np.mean(b))**2)

                desc_parts = []
                if n2o4_corrected:
                    desc_parts.append(fitting_info['n2o4_equilibrium']['description'])
                if n2o5_corrected:
                    desc_parts.append(fitting_info['n2o5_equilibrium']['description'])

                fitting_info['step4_equilibrium'] = {
                    'concentrations': conc.copy(),
                    'r2': r2_after_eq,
                    'description': ' | '.join(desc_parts),
                    'n2o4_corrected': n2o4_corrected,
                    'n2o5_corrected': n2o5_corrected,
                }

        # ===== Detection Limit 적용 =====
        conc, fitting_info = self.apply_detection_limits(conc, fitting_info)

        # ── [v8.14] 전구체 상한 적용: Step4 평형 보정 + DL 적용 후 ────
        # cap된 NO2/N2O5 기준으로 HONO/HONO2 상한 계산 (순서 불일치 해소)
        if self.humidity_weight > 0:
            conc, fitting_info = self.apply_hono_precursor_cap(conc, fitting_info)

        # ===== NO3 피크 제약을 맨 마지막에 적용 (v8.4 최종) =====
        if self.no3_peak_constraint and no3_idx is not None and conc[no3_idx] > 0:
            print(f"\n  [최종 단계] NO3 피크 제약 적용 중...")
            print(f"  [DEBUG] 제약 전 NO3: {conc[no3_idx]:.2e}")
            
            conc = self.apply_no3_peak_constraint(wavelengths, b, A, conc, no3_idx)
            
            print(f"  [DEBUG] 제약 후 NO3: {conc[no3_idx]:.2e}")
            
            # 제약 적용 후 최종 R² 계산
            fitted_after_constraint = A @ conc
            r2_after_constraint = 1 - np.sum((fitted_after_constraint - b)**2) / np.sum((b - np.mean(b))**2)
            
            fitting_info['no3_peak_constraint_final'] = {
                'concentrations': conc.copy(),
                'r2': r2_after_constraint,
                'description': f'NO3 peak constraint (FINAL STEP) ({self.no3_peak_range[0]}-{self.no3_peak_range[1]}nm)'
            }
        
        # 최종 피팅
        fitted_final = A @ conc
        r2_final = 1 - np.sum((fitted_final - b)**2) / np.sum((b - np.mean(b))**2)
        
        return conc, r2_final, fitting_info
    def analyze_single_file_detailed(self, filepath: str) -> Dict:
        """
        단일 파일 상세 분석 (개선된 피팅 + HONO 제거 옵션)
        """
        print("\n" + "="*60)
        print(f"단일 파일 상세 분석")
        print("="*60)
        print(f"파일: {os.path.basename(filepath)}")
        print(f"습도 가중치: {self.humidity_weight:.2f}")

        # 데이터 로드
        abs_data = self.load_data_file(filepath)
        if abs_data is None:
            print("⚠ 파일 로드 실패")
            return None

        # 피팅 범위 선택
        mask = (abs_data['wavelength'] >= self.fitting_range[0]) & \
               (abs_data['wavelength'] <= self.fitting_range[1])
        abs_filtered = abs_data[mask]

        wavelengths_full = abs_filtered['wavelength'].values.copy()
        absorbance_full = abs_filtered['value'].values.copy()
        absorbance_full[absorbance_full < 0] = 0
        
        # === 노이즈 필터링: threshold 이하 값은 0으로 처리 ===
        n_noise_filtered = np.sum(absorbance_full < self.noise_threshold)
        absorbance_full[absorbance_full < self.noise_threshold] = 0
        if n_noise_filtered > 0:
            print(f"✓ 노이즈 필터링: {n_noise_filtered}개 점 → 0 (absorbance < {self.noise_threshold:.0e})")

        # === 전체 파장에 대한 Cross section 매트릭스 (그래프용) ===
        A_full = np.zeros((len(wavelengths_full), len(self.species_list)))
        for i, species in enumerate(self.species_list):
            cs = self.cross_sections[species]
            cs_interp = np.interp(
                wavelengths_full,
                cs['wavelength'].values,
                cs['value'].values,
                left=0, right=0
            )
            A_full[:, i] = cs_interp * self.path_length

        # === 포화 영역 제외 (피팅용) ===
        # [v8.15] 방법 A(threshold) + 방법 B(파장 범위) 동시 지원
        valid_mask = absorbance_full < self.saturation_threshold  # 방법 A

        if self.exclude_wavelength_range is not None:             # 방법 B
            wl_ex1, wl_ex2 = self.exclude_wavelength_range
            wl_range_mask = (wavelengths_full < wl_ex1) | (wavelengths_full > wl_ex2)
            valid_mask = valid_mask & wl_range_mask

        n_excluded = np.sum(~valid_mask)
        if n_excluded > 0:
            excluded_wl_min = wavelengths_full[~valid_mask].min()
            excluded_wl_max = wavelengths_full[~valid_mask].max()
            msg = f"⚠ 포화 영역 제외: {n_excluded}개 점"
            msg += f" (absorbance >= {self.saturation_threshold}"
            if self.exclude_wavelength_range is not None:
                msg += f" + {self.exclude_wavelength_range[0]}-{self.exclude_wavelength_range[1]}nm 고정 제외"
            msg += f")  제외 파장: {excluded_wl_min:.1f}-{excluded_wl_max:.1f} nm"
            print(msg)

        # 피팅용 데이터 (포화 영역 제외)
        wavelengths = wavelengths_full[valid_mask]
        absorbance = absorbance_full[valid_mask]

        print(f"피팅 데이터: {len(wavelengths)} points ({wavelengths.min():.1f}-{wavelengths.max():.1f} nm)")

        # 피팅용 Cross section 매트릭스
        A = np.zeros((len(wavelengths), len(self.species_list)))

        for i, species in enumerate(self.species_list):
            cs = self.cross_sections[species]
            cs_interp = np.interp(
                wavelengths,
                cs['wavelength'].values,
                cs['value'].values,
                left=0, right=0
            )
            A[:, i] = cs_interp * self.path_length

        # 플라즈마 타입 판별
        plasma_info = self.detect_plasma_type(wavelengths, absorbance)
        print(f"\n플라즈마 타입: {plasma_info['type']}")
        if plasma_info['ozone_peak']['has_peak']:
            print(f"  O3 피크: {plasma_info['ozone_peak']['wavelength']:.1f} nm")
            print(f"  O3 비율: {plasma_info['ozone_peak']['ratio']:.1%}")

        # 피팅 옵션 표시
        print(f"\n피팅 옵션:")
        print(f"  NO2 강제 추가 (Step 3): {'ON' if self.force_no2 else 'OFF'}")
        print(f"  Overfitting 보정 (Step 4): {'ON' if self.correct_overfit else 'OFF'}")
        print(f"  N2O4 평형 보정: {'ON' if self.correct_n2o4_equilibrium else 'OFF'}")
        if self.correct_n2o4_equilibrium:
            print(f"    온도: {self.temperature:.1f} K")

        # 개선된 피팅 수행
        print("\n개선된 다단계 피팅 수행 중...")
        final_conc, final_r2, fitting_info = self.improved_nnls_fitting(A, absorbance, wavelengths)

        # 단계별 결과 출력
        print(f"\n피팅 단계별 결과:")
        for step_name, step_info in fitting_info.items():
            if step_name in ('n2o4_equilibrium', 'n2o5_equilibrium'):
                # 이 항목은 step4_equilibrium에서 처리됨
                continue
            if 'r2' in step_info:
                print(f"  {step_name}: {step_info['description']} (R²={step_info['r2']:.4f})")
            else:
                print(f"  {step_name}: {step_info['description']}")
            if 'no2_forced' in step_info and step_info['no2_forced']:
                print(f"    → NO2 강제 추가됨")
            if 'hono_adjusted' in step_info and step_info['hono_adjusted']:
                print(f"    → HONO 조정됨")
            if 'n2o4_corrected' in step_info and step_info['n2o4_corrected']:
                print(f"    → N2O4 평형 보정됨")
            if 'n2o5_corrected' in step_info and step_info['n2o5_corrected']:
                print(f"    → N2O5 평형 보정됨")  # [v8.9]
            # Detection limit 정보 출력
            if step_name == 'detection_limit' and step_info.get('applied', False):
                species_list = list(step_info['species_below_dl'].keys())
                print(f"    → {', '.join(species_list)} (Detection Limit 이하)")

        # 결과 출력
        print(f"\n최종 피팅 결과:")
        print(f"  R² = {final_r2:.4f}")
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                dl = self.detection_limits.get(sp, 0)
                dl_ppm = dl / 2.46e10  # Convert to ppm at 298K, 1atm
                print(f"  {sp}: {final_conc[i]:.3e} (DL: {dl:.1e} = {dl_ppm:.1f} ppm)")
        
        # Detection limit 이하로 제거된 종 표시
        if 'detection_limit' in fitting_info and fitting_info['detection_limit'].get('applied', False):
            below_dl_info = fitting_info['detection_limit']['species_below_dl']
            if below_dl_info:
                print(f"\n⚠ Detection Limit 이하로 0 처리된 화학종:")
                for sp, info in below_dl_info.items():
                    dl_ppm = info['detection_limit'] / 2.46e10
                    print(f"  {sp}: {info['original']:.2e} → 0 (DL: {info['detection_limit']:.1e} = {dl_ppm:.1f} ppm)")

        # HONO/HONO2 제거 옵션 제공 (습도 > 0일 때만)
        if self.humidity_weight > 0:
            hono_idx = self.species_list.index('HONO') if 'HONO' in self.species_list else None
            hono2_idx = self.species_list.index('HONO2') if 'HONO2' in self.species_list else None

            hono_conc = final_conc[hono_idx] if hono_idx is not None else 0
            hono2_conc = final_conc[hono2_idx] if hono2_idx is not None else 0

            # HONO 또는 HONO2가 상당한 농도를 가지면 옵션 제공
            if hono_conc > 1e-10 or hono2_conc > 1e-10:
                print("\n" + "="*60)
                print("HONO/HONO2 재피팅 옵션")
                print("="*60)
                print(f"\n현재 HONO/HONO2 농도:")
                if hono_conc > 1e-10:
                    print(f"  HONO:  {hono_conc:.3e}")
                if hono2_conc > 1e-10:
                    print(f"  HONO2: {hono2_conc:.3e}")

                print("\nHONO/HONO2 값이 크게 나타났습니다.")
                print("이들을 제외하고 다시 피팅하시겠습니까?")
                print("\n0: 현재 결과 유지")
                print("1: HONO만 제거")
                print("2: HONO2만 제거")
                print("3: HONO와 HONO2 모두 제거")

                while True:
                    try:
                        choice = input("\n선택 (0-3): ").strip()
                        if choice in ['0', '1', '2', '3']:
                            choice = int(choice)
                            break
                        else:
                            print("⚠ 0, 1, 2, 또는 3을 입력해주세요.")
                    except KeyboardInterrupt:
                        print("\n\n현재 결과를 유지합니다.")
                        choice = 0
                        break

                if choice > 0:
                    # A 매트릭스 수정
                    A_modified = A.copy()
                    excluded_species = []

                    if choice == 1 or choice == 3:
                        if hono_idx is not None:
                            A_modified[:, hono_idx] = 0
                            excluded_species.append('HONO')

                    if choice == 2 or choice == 3:
                        if hono2_idx is not None:
                            A_modified[:, hono2_idx] = 0
                            excluded_species.append('HONO2')

                    print(f"\n{', '.join(excluded_species)} 제외하여 재피팅...")

                    # 재피팅 수행 (처음부터 다시)
                    final_conc_new, final_r2_new, fitting_info_new = self.improved_nnls_fitting(
                        A_modified, absorbance, wavelengths
                    )

                    # 제외된 종은 강제로 0
                    if choice == 1 or choice == 3:
                        if hono_idx is not None:
                            final_conc_new[hono_idx] = 0
                    if choice == 2 or choice == 3:
                        if hono2_idx is not None:
                            final_conc_new[hono2_idx] = 0

                    print(f"\n재피팅 결과:")
                    print(f"  R² = {final_r2_new:.4f} (이전: {final_r2:.4f})")

                    for i, sp in enumerate(self.species_list):
                        if final_conc_new[i] > 1e-10:
                            old_val = final_conc[i]
                            new_val = final_conc_new[i]
                            if old_val > 1e-10:
                                change = (new_val - old_val) / old_val * 100
                                print(f"  {sp}: {new_val:.3e} (변화: {change:+.1f}%)")
                            else:
                                print(f"  {sp}: {new_val:.3e} (새로 검출)")

                    # 업데이트된 결과 사용
                    final_conc = final_conc_new
                    final_r2 = final_r2_new
                    fitting_info = fitting_info_new

                    # 피팅 단계에 재피팅 정보 추가
                    fitting_info['refit'] = {
                        'concentrations': final_conc_new.copy(),
                        'r2': final_r2_new,
                        'description': f'{", ".join(excluded_species)} excluded',
                        'excluded': excluded_species
                    }

        # 피팅 단계 정보 구성 (피팅용 파장 기준)
        fitting_stages = {}
        for step_name, step_info in fitting_info.items():
            if 'concentrations' not in step_info:
                continue  # concentrations가 없는 단계는 건너뜀
            fitted = A @ step_info['concentrations']
            fitting_stages[step_name] = {
                'concentrations': step_info['concentrations'],
                'fitted': fitted,
                'r2': step_info.get('r2', 0),
                'description': step_info.get('description', '')
            }

        # 전체 파장에 대한 피팅 결과 (그래프용)
        fitting_stages_full = {}
        for step_name, step_info in fitting_info.items():
            if 'concentrations' not in step_info:
                continue  # concentrations가 없는 단계는 건너뜀
            fitted_full = A_full @ step_info['concentrations']
            fitting_stages_full[step_name] = {
                'concentrations': step_info['concentrations'],
                'fitted': fitted_full,
                'r2': step_info.get('r2', 0),
                'description': step_info.get('description', '')
            }

        return {
            'wavelengths': wavelengths,
            'absorbance': absorbance,
            'wavelengths_full': wavelengths_full,
            'absorbance_full': absorbance_full,
            'valid_mask': valid_mask,
            'fitting_stages': fitting_stages,
            'fitting_stages_full': fitting_stages_full,
            'plasma_info': plasma_info,
            'final_concentrations': final_conc,
            'final_r2': final_r2,
            'A_matrix': A,
            'A_matrix_full': A_full,
            'n_saturated_excluded': n_excluded
        }

    def analyze_single_file(self, filepath: str, time_point: float) -> Dict:
        """
        단일 파일 분석 (배치용, 개선된 피팅)

        [v8.12] 온도 프로파일 연동 버그 수정
        - 온도 파일이 입력된 경우, time_point에 해당하는 온도를 조회하여
          self.temperature를 임시 갱신한 뒤 평형 보정(Keq)에 반영
        - 피팅 완료 후 원래 온도로 복원 (다른 파일에 영향 없음)
        - 온도 파일 없는 경우: self.temperature = 298 K 고정 → 동작 변화 없음
        """
        # === [v8.12] 파일별 온도 갱신 (온도 프로파일 있을 때만) ===
        _temperature_backup = self.temperature          # 복원용 백업
        if self.temperature_profile is not None:
            self.temperature = self.get_temperature_at_time(time_point)

        # 데이터 로드
        abs_data = self.load_data_file(filepath)
        if abs_data is None:
            return None

        # 피팅 범위 선택
        mask = (abs_data['wavelength'] >= self.fitting_range[0]) & \
               (abs_data['wavelength'] <= self.fitting_range[1])
        abs_filtered = abs_data[mask]

        wavelengths_full = abs_filtered['wavelength'].values.copy()
        absorbance_full = abs_filtered['value'].values.copy()
        absorbance_full[absorbance_full < 0] = 0
        
        # === 노이즈 필터링: threshold 이하 값은 0으로 처리 ===
        absorbance_full[absorbance_full < self.noise_threshold] = 0

        # === 전체 파장에 대한 Cross section 매트릭스 (그래프용) ===
        A_full = np.zeros((len(wavelengths_full), len(self.species_list)))
        for i, species in enumerate(self.species_list):
            cs = self.cross_sections[species]
            cs_interp = np.interp(
                wavelengths_full,
                cs['wavelength'].values,
                cs['value'].values,
                left=0, right=0
            )
            A_full[:, i] = cs_interp * self.path_length

        # === 포화 영역 제외 (피팅용) ===
        valid_mask = absorbance_full < self.saturation_threshold  # 방법 A
        if self.exclude_wavelength_range is not None:             # 방법 B
            wl_ex1, wl_ex2 = self.exclude_wavelength_range
            valid_mask = valid_mask & ((wavelengths_full < wl_ex1) | (wavelengths_full > wl_ex2))
        n_excluded = np.sum(~valid_mask)
        if n_excluded > 0:
            print(f"  ⚠ 포화 영역 제외: {n_excluded}개 점 (threshold={self.saturation_threshold}"
                  + (f", {self.exclude_wavelength_range[0]}-{self.exclude_wavelength_range[1]}nm"
                     if self.exclude_wavelength_range else "") + ")")

        wavelengths = wavelengths_full[valid_mask]
        absorbance = absorbance_full[valid_mask]

        # 피팅용 Cross section 매트릭스
        A = np.zeros((len(wavelengths), len(self.species_list)))

        for i, species in enumerate(self.species_list):
            cs = self.cross_sections[species]
            cs_interp = np.interp(
                wavelengths,
                cs['wavelength'].values,
                cs['value'].values,
                left=0, right=0
            )
            A[:, i] = cs_interp * self.path_length

        # 플라즈마 타입 판별
        plasma_info = self.detect_plasma_type(wavelengths, absorbance)

        # 개선된 피팅 수행
        concentrations, r2, fitting_info = self.improved_nnls_fitting(A, absorbance, wavelengths)

        # 전체 파장에 대한 피팅 결과 계산
        fitted_full = A_full @ concentrations
        
        # N2O4 평형 정보 추출
        n2o4_eq_info = fitting_info.get('n2o4_equilibrium', {})

        # === [v8.12] 온도 복원 (다른 파일 분석에 영향 없도록) ===
        self.temperature = _temperature_backup

        return {
            'time': time_point,
            'concentrations': concentrations,
            'r2': r2,
            'plasma_type': plasma_info['type'],
            'filename': os.path.basename(filepath),
            'n_saturated_excluded': n_excluded,
            'n2o4_equilibrium_info': n2o4_eq_info,
            # 그래프용 추가 데이터
            'wavelengths_full': wavelengths_full,
            'absorbance_full': absorbance_full,
            'valid_mask': valid_mask,
            'fitted_full': fitted_full,
            'A_full': A_full
        }

    def plot_batch_single_result(self, result: Dict, filepath: str, png_output_dir: str):
        """
        배치 분석 시 각 파일에 대한 종합 플롯 생성 (로그 스케일)
        """
        # 데이터 추출
        wavelengths_full = result['wavelengths_full']
        absorbance_full = result['absorbance_full']
        valid_mask = result['valid_mask']
        fitted_full = result['fitted_full']
        A_full = result['A_full']
        final_conc = result['concentrations']
        
        # Measured 데이터 준비 (포화 영역은 NaN)
        absorbance_plot = absorbance_full.copy().astype(float)
        absorbance_plot[~valid_mask] = np.nan

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(16, 10))

        # 색상 팔레트 정의
        species_colors = {
            'O3': '#1f77b4',      # 파랑
            'NO': '#ff7f0e',      # 주황
            'NO2': '#2ca02c',     # 초록
            'NO3': '#d62728',     # 빨강
            'N2O4': '#9467bd',    # 보라
            'N2O5': '#8c564b',    # 갈색
            'HONO': '#e377c2',    # 분홍
            'HONO2': '#7f7f7f'    # 회색
        }

        # Measured 데이터 (dot 그래프, 포화 영역 제외)
        abs_positive_plot = np.where(absorbance_plot > 0, absorbance_plot, np.nan)
        ax.semilogy(wavelengths_full, abs_positive_plot, 'ko', 
                    markersize=5, alpha=0.4, label='Measured', zorder=1)

        # 개별 종 기여도
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                contribution = A_full[:, i] * final_conc[i]
                contribution_positive = np.where(contribution > 0, contribution, np.nan)
                color = species_colors.get(sp, f'C{i}')
                
                if sp in ['HONO', 'HONO2']:
                    ax.semilogy(wavelengths_full, contribution_positive, '--', 
                                color=color, alpha=0.7, linewidth=1.5,
                                label=f'{sp}: {final_conc[i]:.2e}')
                else:
                    ax.semilogy(wavelengths_full, contribution_positive, '-', 
                                color=color, alpha=0.8, linewidth=1.5,
                                label=f'{sp}: {final_conc[i]:.2e}')

        # Fitted 결과
        fitted_positive_full = np.where(fitted_full > 0, fitted_full, np.nan)
        ax.semilogy(wavelengths_full, fitted_positive_full, 'r-', 
                    linewidth=2.5, alpha=0.9, label=f'Fitted (R²={result["r2"]:.4f})', zorder=10)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Absorbance (log scale)', fontsize=12)
        ax.set_title(f'Batch Analysis - {result["filename"]} (Time: {result["time"]})\n'
                     f'Measured (dots) + Fitted (red line) + Individual Species Contributions',
                     fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([wavelengths_full.min(), wavelengths_full.max()])
        ax.set_ylim([1e-6, 20]) # y축 고정

        plt.tight_layout()

        # 저장
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        base_name = os.path.splitext(result['filename'])[0]
        output_file = os.path.join(png_output_dir,
                                   f'{base_name}_analysis_{humidity_str}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()  # 실시간으로 플롯 표시

        print(f"  ✓ 그래프 저장: {os.path.basename(output_file)}")

    def batch_analyze(self, folder_path: str, plot_each: bool = True):
        """
        폴더 내 모든 파일 배치 분석
        
        Parameters:
        -----------
        folder_path : str
            분석할 파일들이 있는 폴더 경로
        plot_each : bool
            각 파일별 분석 플롯 생성 여부 (기본값: True)
        """
        # === MODIFIED: Absorbance + __<정수>__ 패턴을 만족하는 파일만 ===
        all_txt = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
        files = []
        skipped = []
        for f in all_txt:
            if ('absorbance' in f.lower()) and self.TIME_TOKEN.search(f):
                files.append(f)
            elif ('absorbance' in f.lower()):
                skipped.append(f)

        # 시간 순으로 정렬
        file_time_pairs = []
        for f in files:
            time_point = self.extract_time_from_filename(f)
            if time_point is not None:
                file_time_pairs.append((f, time_point))

        file_time_pairs.sort(key=lambda x: x[1])

        if not file_time_pairs:
            print("⚠ 분석할 Absorbance 파일이 없습니다. (조건: 'Absorbance' + '__<정수>__')")
            return

        if skipped:
            print("\n⚠ 시간 토큰(__<정수>__)이 없어 제외된 파일:")
            for s in skipped[:5]:
                print("   -", s)
            if len(skipped) > 5:
                print(f"   ... 외 {len(skipped)-5}개")

        # === PNG 결과 폴더 생성 (plot_each가 True일 때만) ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        png_folder_name = f"png result_{humidity_str}_{timestamp}"
        png_output_dir = os.path.join(folder_path, png_folder_name)
        
        if plot_each:
            os.makedirs(png_output_dir, exist_ok=True)
        
        print(f"\n총 {len(file_time_pairs)}개 파일 분석 시작...")
        print(f"습도 가중치: {self.humidity_weight:.2f}")
        print(f"포화 영역 제외: absorbance >= {self.saturation_threshold}")
        if plot_each:
            print(f"개별 파일 플롯: ON (저장 폴더: {png_folder_name})")
        else:
            print(f"개별 파일 플롯: OFF (최종 요약 플롯만 생성)")
        print("개선된 피팅 알고리즘 사용")
        print("-" * 50)

        # 결과 저장
        results_list = []

        # 진행 상황 표시
        for i, (filename, time_point) in enumerate(file_time_pairs, 1):
            filepath = os.path.join(folder_path, filename)

            # [v8.8] 시간별 온도 동적 업데이트
            T_at_t = self.get_temperature_at_time(time_point)
            self.temperature = T_at_t
            T_C = T_at_t - 273.15

            print(f"\n[{i}/{len(file_time_pairs)}] 분석 중: {filename}")
            print(f"  시간: {time_point} s  |  온도: {T_C:.1f} °C ({T_at_t:.1f} K)"
                  f"{'  [프로파일]' if self.temperature_profile else '  [고정]'}")

            result = self.analyze_single_file(filepath, time_point)

            result = self.analyze_single_file(filepath, time_point)

            if result:
                result['temperature_K'] = T_at_t   # [v8.8] 온도 기록

                # ── [v8.10] 인접 시점 NO3 제약 (자동 허용 범위) ──────
                if len(results_list) >= 2:
                    no3_idx_t = self.species_list.index('NO3') if 'NO3' in self.species_list else None
                    if no3_idx_t is not None:
                        # 직전 2개 시점의 NO3로 기준값 계산
                        prev_no3_vals = [r['concentrations'][no3_idx_t]
                                         for r in results_list[-2:] if r['concentrations'][no3_idx_t] > 0]
                        if prev_no3_vals:
                            ref_no3   = np.mean(prev_no3_vals)
                            curr_no3  = result['concentrations'][no3_idx_t]

                            if ref_no3 > 0 and curr_no3 > 0:
                                # 자동 허용 범위: 직전 시점들의 표준편차 기반
                                all_no3_so_far = [r['concentrations'][no3_idx_t]
                                                  for r in results_list
                                                  if r['concentrations'][no3_idx_t] > 0]
                                if len(all_no3_so_far) >= 3:
                                    # 변동계수(CV) 기반: 전체 NO3의 상대 표준편차를 허용 범위로
                                    cv = np.std(all_no3_so_far) / np.mean(all_no3_so_far)
                                    # CV를 허용 범위로 사용 (최소 30%, 최대 200%)
                                    allowed_ratio = np.clip(cv * 2, 0.3, 2.0)
                                else:
                                    allowed_ratio = 1.0  # 초기엔 ±100% 허용

                                upper = ref_no3 * (1 + allowed_ratio)
                                lower = ref_no3 * max(0, 1 - allowed_ratio)

                                if curr_no3 > upper:
                                    result['concentrations'][no3_idx_t] = upper
                                    print(f"  [v8.10] NO3 시간축 상한 제약: "
                                          f"{curr_no3:.2e} → {upper:.2e} "
                                          f"(허용범위 ±{allowed_ratio*100:.0f}%)")
                                elif curr_no3 < lower and lower > 0:
                                    result['concentrations'][no3_idx_t] = lower
                                    print(f"  [v8.10] NO3 시간축 하한 제약: "
                                          f"{curr_no3:.2e} → {lower:.2e} "
                                          f"(허용범위 ±{allowed_ratio*100:.0f}%)")

                results_list.append(result)
                print(f"  R² = {result['r2']:.4f}")
                print(f"  플라즈마 타입: {result['plasma_type']}")

                # 주요 농도 출력
                conc = result['concentrations']
                for j, sp in enumerate(self.species_list):
                    if conc[j] > 1e-10:
                        print(f"    {sp}: {conc[j]:.2e}")
                
                # N2O4 / N2O5 평형 보정 정보 출력 [v8.9]
                n2o4_eq_info = result.get('n2o4_equilibrium_info', {})
                if n2o4_eq_info.get('correction_applied', False):
                    print(f"    → N2O4 평형 보정: {n2o4_eq_info.get('ratio_before', 0):.1f}x → 1.0x")
                n2o5_info = result.get('fitting_info', {}).get('n2o5_equilibrium', {})
                if n2o5_info.get('correction_applied', False):
                    print(f"    → N2O5 평형 보정: {n2o5_info.get('ratio_before', 0):.1f}x → 1.0x")

                # === 종합 플롯 생성 (로그 스케일) - 옵션에 따라 ===
                if plot_each:
                    self.plot_batch_single_result(result, filepath, png_output_dir)
            else:
                print("  ⚠ 분석 실패")

        self.batch_results = results_list
        print("\n" + "="*60)
        print(f"배치 분석 완료: {len(results_list)}개 성공")
        if plot_each:
            print(f"PNG 저장 위치: {png_output_dir}")
        else:
            print("개별 파일 플롯: 생성하지 않음")
        print("="*60)

        # HONO/HONO2 평균 농도 계산 및 재분석 옵션 제공
        if self.humidity_weight > 0 and len(results_list) > 0:
            hono_idx = self.species_list.index('HONO') if 'HONO' in self.species_list else None
            hono2_idx = self.species_list.index('HONO2') if 'HONO2' in self.species_list else None

            # 평균 농도 계산
            hono_concentrations = []
            hono2_concentrations = []

            for result in results_list:
                if hono_idx is not None:
                    hono_concentrations.append(result['concentrations'][hono_idx])
                if hono2_idx is not None:
                    hono2_concentrations.append(result['concentrations'][hono2_idx])

            hono_mean = np.mean(hono_concentrations) if hono_concentrations else 0
            hono2_mean = np.mean(hono2_concentrations) if hono2_concentrations else 0

            if hono_mean > 1e-10 or hono2_mean > 1e-10:
                print("\n" + "="*60)
                print("HONO/HONO2 재분석 옵션")
                print("="*60)
                print(f"\n배치 분석 결과 - HONO/HONO2 평균 농도:")
                if hono_mean > 1e-10:
                    print(f"  HONO 평균:  {hono_mean:.3e}")
                    print(f"  HONO 최대:  {np.max(hono_concentrations):.3e}")
                    print(f"  HONO 최소:  {np.min([c for c in hono_concentrations if c > 0] or [0]):.3e}")
                if hono2_mean > 1e-10:
                    print(f"  HONO2 평균: {hono2_mean:.3e}")
                    print(f"  HONO2 최대: {np.max(hono2_concentrations):.3e}")
                    print(f"  HONO2 최소: {np.min([c for c in hono2_concentrations if c > 0] or [0]):.3e}")

                print("\nHONO/HONO2를 제외하고 다시 배치 분석을 수행하시겠습니까?")
                print("\n0: 현재 결과 유지")
                print("1: HONO만 제거하고 재분석")
                print("2: HONO2만 제거하고 재분석")
                print("3: HONO와 HONO2 모두 제거하고 재분석")

                while True:
                    try:
                        choice = input("\n선택 (0-3): ").strip()
                        if choice in ['0', '1', '2', '3']:
                            choice = int(choice)
                            break
                        else:
                            print("⚠ 0, 1, 2, 또는 3을 입력해주세요.")
                    except KeyboardInterrupt:
                        print("\n\n현재 결과를 유지합니다.")
                        choice = 0
                        break

                if choice > 0:
                    # 재분석 수행
                    exclude_hono = (choice == 1 or choice == 3)
                    exclude_hono2 = (choice == 2 or choice == 3)

                    excluded = []
                    if exclude_hono:
                        excluded.append("HONO")
                    if exclude_hono2:
                        excluded.append("HONO2")

                    print(f"\n{', '.join(excluded)} 제외하여 재분석 시작...")
                    print("-" * 50)

                    # 재분석 결과 저장
                    reanalysis_results = []

                    for i, (filename, time_point) in enumerate(file_time_pairs, 1):
                        filepath = os.path.join(folder_path, filename)
                        print(f"\n[{i}/{len(file_time_pairs)}] 재분석 중: {filename}")
                        print(f"  시간: {time_point}")

                        # 재분석 수행
                        result = self.analyze_single_file_with_exclusion(
                            filepath, time_point, exclude_hono, exclude_hono2
                        )

                        if result:
                            reanalysis_results.append(result)
                            print(f"  R² = {result['r2']:.4f}")

                            # 주요 농도 출력 (제외된 종 빼고)
                            conc = result['concentrations']
                            for j, sp in enumerate(self.species_list):
                                if conc[j] > 1e-10:
                                    if sp == 'HONO' and exclude_hono:
                                        continue
                                    if sp == 'HONO2' and exclude_hono2:
                                        continue
                                    print(f"    {sp}: {conc[j]:.2e}")
                        else:
                            print("  ⚠ 재분석 실패")

                    # 재분석 결과로 업데이트
                    self.batch_results = reanalysis_results

                    print("\n" + "="*60)
                    print(f"재분석 완료: {len(reanalysis_results)}개 성공")
                    print(f"제외된 화학종: {', '.join(excluded)}")

                    # R² 비교
                    original_r2 = np.mean([r['r2'] for r in results_list])
                    new_r2 = np.mean([r['r2'] for r in reanalysis_results])
                    print(f"평균 R² 변화: {original_r2:.4f} → {new_r2:.4f} ({(new_r2-original_r2)/original_r2*100:+.1f}%)")
                    print("="*60)

    def analyze_single_file_with_exclusion(self, filepath: str, time_point: float,
                                          exclude_hono: bool = False,
                                          exclude_hono2: bool = False) -> Dict:
        """
        단일 파일 분석 (배치용, HONO/HONO2 제거 옵션 포함)
        """
        # 데이터 로드
        abs_data = self.load_data_file(filepath)
        if abs_data is None:
            return None

        # 피팅 범위 선택
        mask = (abs_data['wavelength'] >= self.fitting_range[0]) & \
               (abs_data['wavelength'] <= self.fitting_range[1])
        abs_filtered = abs_data[mask]

        wavelengths = abs_filtered['wavelength'].values
        absorbance = abs_filtered['value'].values
        absorbance[absorbance < 0] = 0

        # === 추가: 포화 영역 제외 (absorbance >= saturation_threshold) ===
        valid_mask = absorbance < self.saturation_threshold
        n_excluded = np.sum(~valid_mask)
        if n_excluded > 0:
            print(f"  ⚠ 포화 영역 제외: {n_excluded}개 점 (absorbance >= {self.saturation_threshold})")
            wavelengths = wavelengths[valid_mask]
            absorbance = absorbance[valid_mask]

        # Cross section 매트릭스 생성
        A = np.zeros((len(wavelengths), len(self.species_list)))

        for i, species in enumerate(self.species_list):
            cs = self.cross_sections[species]
            cs_interp = np.interp(
                wavelengths,
                cs['wavelength'].values,
                cs['value'].values,
                left=0, right=0
            )
            A[:, i] = cs_interp * self.path_length

        # HONO/HONO2 제거 옵션 적용
        if exclude_hono and 'HONO' in self.species_list:
            hono_idx = self.species_list.index('HONO')
            A[:, hono_idx] = 0

        if exclude_hono2 and 'HONO2' in self.species_list:
            hono2_idx = self.species_list.index('HONO2')
            A[:, hono2_idx] = 0

        # 플라즈마 타입 판별
        plasma_info = self.detect_plasma_type(wavelengths, absorbance)

        # 개선된 피팅 수행
        concentrations, r2, _ = self.improved_nnls_fitting(A, absorbance, wavelengths)

        # 제거된 종은 강제로 0
        if exclude_hono and 'HONO' in self.species_list:
            hono_idx = self.species_list.index('HONO')
            concentrations[hono_idx] = 0

        if exclude_hono2 and 'HONO2' in self.species_list:
            hono2_idx = self.species_list.index('HONO2')
            concentrations[hono2_idx] = 0

        return {
            'time': time_point,
            'concentrations': concentrations,
            'r2': r2,
            'plasma_type': plasma_info['type'],
            'filename': os.path.basename(filepath),
            'n_saturated_excluded': n_excluded  # 추가: 제외된 포화점 수
        }

    def save_batch_results(self, folder_path: str):
        """
        배치 분석 결과 저장
        """
        if not self.batch_results:
            print("저장할 결과가 없습니다.")
            return

        # 시간별 농도 데이터프레임 생성
        time_points = []
        species_data = {sp: [] for sp in self.species_list}
        r2_values = []
        plasma_types = []
        saturated_excluded = []  # 추가
        temperature_K_list = []  # [v8.8]

        for result in self.batch_results:
            time_points.append(result['time'])
            r2_values.append(result['r2'])
            plasma_types.append(result['plasma_type'])
            saturated_excluded.append(result.get('n_saturated_excluded', 0))  # 추가
            temperature_K_list.append(result.get('temperature_K', self.temperature))  # [v8.8]

            for i, sp in enumerate(self.species_list):
                species_data[sp].append(result['concentrations'][i])

        # 데이터프레임 생성
        df = pd.DataFrame({
            'Time': time_points,
            'Temperature_K': temperature_K_list,                                  # [v8.8]
            'Temperature_C': [T - 273.15 for T in temperature_K_list],           # [v8.8]
            **species_data,
            'R2': r2_values,
            'Plasma_Type': plasma_types,
            'Humidity_Weight': [self.humidity_weight] * len(time_points),
            'Saturated_Excluded': saturated_excluded  # 추가
        })

        # 시간순 정렬
        df = df.sort_values('Time').reset_index(drop=True)

        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        output_file = os.path.join(folder_path, f'batch_results_improved_{humidity_str}_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f"\n✓ 결과 저장: {output_file}")

        # 요약 통계 출력
        print("\n" + "="*60)
        print("요약 통계")
        print("="*60)

        for sp in self.species_list:
            conc_arr = df[sp].values
            non_zero = conc_arr[conc_arr > 1e-10]
            if len(non_zero) > 0:
                print(f"\n{sp}:")
                print(f"  평균: {np.mean(non_zero):.3e}")
                print(f"  최대: {np.max(non_zero):.3e}")
                print(f"  최소: {np.min(non_zero):.3e}")
                print(f"  검출: {len(non_zero)}/{len(conc_arr)} 파일")

        print(f"\n평균 R²: {df['R2'].mean():.4f}")

        # 포화 영역 제외 통계 추가
        total_excluded = df['Saturated_Excluded'].sum()
        files_with_excluded = (df['Saturated_Excluded'] > 0).sum()
        if total_excluded > 0:
            print(f"\n포화 영역 제외 통계:")
            print(f"  총 제외 점: {total_excluded}개")
            print(f"  영향받은 파일: {files_with_excluded}/{len(df)}개")

        return df

    def plot_single_file_results(self, result: Dict, filepath: str):
        """
        단일 파일 분석 결과 시각화 (개선된 버전)
        - Measured: 포화 영역 제외되어 끊어짐
        - Fitted: 전체 파장에서 연속적으로 표시
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # 피팅에 사용된 데이터 (포화 영역 제외)
        wavelengths = result['wavelengths']
        absorbance = result['absorbance']
        stages = result['fitting_stages']
        A = result['A_matrix']

        # 전체 파장 데이터 (그래프용)
        wavelengths_full = result.get('wavelengths_full', wavelengths)
        absorbance_full = result.get('absorbance_full', absorbance)
        stages_full = result.get('fitting_stages_full', stages)
        A_full = result.get('A_matrix_full', A)
        valid_mask = result.get('valid_mask', np.ones(len(wavelengths_full), dtype=bool))

        # 최종 단계 찾기
        if 'step4' in stages:
            final_stage = 'step4'
        elif 'step3' in stages:
            final_stage = 'step3'
        elif 'step2' in stages:
            final_stage = 'step2'
        else:
            final_stage = 'step1'

        final_conc = result['final_concentrations']



        # === Measured 데이터 준비 (포화 영역은 NaN으로 표시) ===
        absorbance_plot = absorbance_full.copy().astype(float)
        absorbance_plot[~valid_mask] = np.nan

        # 1. 원본 스펙트럼과 최종 피팅 (로그 스케일)
        ax = axes[0, 0]
        # Measured: 포화 영역 제외 (끊어짐)
        abs_positive = np.where(absorbance_plot > 0, absorbance_plot, np.nan)
        ax.semilogy(wavelengths_full, abs_positive, 'k-', alpha=0.5, label='Measured', linewidth=1)

        # Fitted: 전체 파장에서 연속
        fitted_full = stages_full[final_stage]['fitted']
        fitted_positive = np.where(fitted_full > 0, fitted_full, np.nan)
        ax.semilogy(wavelengths_full, fitted_positive, 'r-',
                    label=f"Fitted (R²={stages[final_stage]['r2']:.4f})", linewidth=2)

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance (log scale)')
        ax.set_title('Final Fit (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 잔차 (피팅에 사용된 영역만)
        ax = axes[0, 1]
        residual = absorbance - stages[final_stage]['fitted']
        wl_plot, res_plot = insert_nan_at_gaps(wavelengths, residual)
        ax.plot(wl_plot, res_plot, 'g-', alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Residual')
        ax.set_title('Fitting Residual (fitted region only)')
        ax.grid(True, alpha=0.3)

        # 3. 단계별 피팅 비교 (로그 스케일) - 전체 파장
        ax = axes[0, 2]
        colors = ['blue', 'green', 'red', 'purple']
        for i, (stage_name, stage_data) in enumerate(stages_full.items()):
            if i < len(colors):
                fitted_positive = np.where(stage_data['fitted'] > 0,
                                           stage_data['fitted'], np.nan)
                ax.semilogy(wavelengths_full, fitted_positive,
                            color=colors[i], alpha=0.7,
                            label=f"{stage_data['description']} (R²={stage_data['r2']:.4f})",
                            linewidth=1.5)

        # Measured (포화 영역 제외)
        ax.semilogy(wavelengths_full, abs_positive, 'k--', alpha=0.3, label='Measured', linewidth=1)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance (log scale)')
        ax.set_title('Fitting Stages Comparison')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. 농도 막대 그래프
        ax = axes[1, 0]
        non_zero_species = [(sp, final_conc[i]) for i, sp in enumerate(self.species_list)
                            if final_conc[i] > 1e-10]

        if non_zero_species:
            species_names, conc_values = zip(*non_zero_species)
            colors_bar = ['blue' if sp == 'O3' else 'red' if 'NO' in sp else 'green'
                          for sp in species_names]

            bars = ax.bar(range(len(species_names)), conc_values, color=colors_bar, alpha=0.7)
            ax.set_xticks(range(len(species_names)))
            ax.set_xticklabels(species_names, rotation=45)
            ax.set_ylabel('Concentration')
            ax.set_yscale('log')
            ax.set_title('Final Concentrations')
            ax.grid(True, alpha=0.3, axis='y')

            # 값 표시
            for bar, val in zip(bars, conc_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2e}', ha='center', va='bottom', fontsize=8)

        # 5. 개별 종 기여도 (선형 스케일) - 전체 파장
        ax = axes[1, 1]
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                contribution = A_full[:, i] * final_conc[i]

                if sp in ['HONO', 'HONO2']:
                    ax.plot(wavelengths_full, contribution, '--',
                            label=f'{sp} (H)', alpha=0.5, linewidth=1)
                else:
                    ax.plot(wavelengths_full, contribution, '-',
                            label=sp, alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance Contribution')
        ax.set_title('Individual Species Contributions (Linear)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6. 개별 종 기여도 (로그 스케일) - 전체 파장
        ax = axes[1, 2]
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                contribution = A_full[:, i] * final_conc[i]
                contribution_positive = np.where(contribution > 0, contribution, np.nan)

                if sp in ['HONO', 'HONO2']:
                    ax.semilogy(wavelengths_full, contribution_positive, '--',
                                label=f'{sp} (H)', alpha=0.5, linewidth=1)
                else:
                    ax.semilogy(wavelengths_full, contribution_positive, '-',
                                label=sp, alpha=0.7, linewidth=1.5)

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance Contribution (log)')
        ax.set_title('Individual Species Contributions (Log Scale)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 7. NO2 영역 확대 - 전체 파장 기준
        ax = axes[2, 0]
        no2_range_full = (wavelengths_full >= 360) & (wavelengths_full <= 410)
        if np.any(no2_range_full):
            wl_no2 = wavelengths_full[no2_range_full]
            
            # Measured (포화 영역은 NaN)
            abs_no2 = absorbance_plot[no2_range_full]
            ax.plot(wl_no2, abs_no2, 'k-', alpha=0.7, label='Measured', linewidth=2)
            
            # Fitted (전체 연속)
            fitted_no2 = stages_full[final_stage]['fitted'][no2_range_full]
            ax.plot(wl_no2, fitted_no2, 'r-', alpha=0.7, label='Fitted', linewidth=2)

            # NO2 기여도만 표시 (전체 연속)
            if 'NO2' in self.species_list:
                no2_idx = self.species_list.index('NO2')
                if final_conc[no2_idx] > 0:
                    no2_contrib = A_full[no2_range_full, no2_idx] * final_conc[no2_idx]
                    ax.plot(wl_no2, no2_contrib, 'b--', alpha=0.7,
                            label=f'NO2 ({final_conc[no2_idx]:.2e})', linewidth=1.5)

            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Absorbance')
            ax.set_title('NO2 Region (360-410 nm)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 8. 피팅 단계 비교 (막대)
        ax = axes[2, 1]
        stage_names = list(stages.keys())
        stage_r2s = [stages[s]['r2'] for s in stage_names]

        bars = ax.bar(range(len(stage_names)), stage_r2s,
                      color=['blue', 'green', 'red', 'purple'][:len(stage_names)], alpha=0.7)
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels([stages[s]['description'] for s in stage_names],
                           rotation=45, ha='right')
        ax.set_ylabel('R²')
        ax.set_title('Fitting Quality by Stage')
        ax.set_ylim([0.9 * min(stage_r2s), 1])
        ax.grid(True, alpha=0.3, axis='y')

        for bar, r2 in zip(bars, stage_r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{r2:.4f}', ha='center', va='bottom')

        # 9. 정보 텍스트
        ax = axes[2, 2]
        ax.axis('off')

        info_text = f"""
=== 분석 정보 ===

파일: {os.path.basename(filepath)}

플라즈마 타입: {result['plasma_info']['type']}

습도 설정: {self.humidity_weight*100:.0f}%

최종 R²: {result['final_r2']:.4f}

=== 주요 농도 ===
"""

        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                info_text += f"\n{sp}: {final_conc[i]:.3e}"

        # NO2 강제 여부 확인
        if 'step2' in stages and 'no2_forced' in stages:
            info_text += "\n\n* NO2 강제 피팅 적용됨"

        ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace')

        plt.suptitle(f'Single File Analysis - Improved Fitting (Humidity: {self.humidity_weight*100:.0f}%)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        output_dir = os.path.dirname(filepath)
        output_file = os.path.join(output_dir,
                                   f'single_analysis_improved_{humidity_str}_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n✓ 그래프 저장: {output_file}")

        # ===============================================
        # 10. 종합 플롯 (로그 스케일 버전)
        # ===============================================
        fig3, ax3 = plt.subplots(figsize=(16, 10))

        # 색상 팔레트 정의
        species_colors = {
            'O3': '#1f77b4',      # 파랑
            'NO': '#ff7f0e',      # 주황
            'NO2': '#2ca02c',     # 초록
            'NO3': '#d62728',     # 빨강
            'N2O4': '#9467bd',    # 보라
            'N2O5': '#8c564b',    # 갈색
            'HONO': '#e377c2',    # 분홍
            'HONO2': '#7f7f7f'    # 회색
        }

        # Measured 데이터 (dot 그래프, 포화 영역 제외) - dot 크기 2.5배 증가
        abs_positive_plot = np.where(absorbance_plot > 0, absorbance_plot, np.nan)
        ax3.semilogy(wavelengths_full, abs_positive_plot, 'ko', 
                     markersize=5, alpha=0.4, label='Measured', zorder=1)

        # 개별 종 기여도
        fitted_full = stages_full[final_stage]['fitted']
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                contribution = A_full[:, i] * final_conc[i]
                contribution_positive = np.where(contribution > 0, contribution, np.nan)
                color = species_colors.get(sp, f'C{i}')
                
                if sp in ['HONO', 'HONO2']:
                    ax3.semilogy(wavelengths_full, contribution_positive, '--', 
                                 color=color, alpha=0.7, linewidth=1.5,
                                 label=f'{sp}: {final_conc[i]:.2e}')
                else:
                    ax3.semilogy(wavelengths_full, contribution_positive, '-', 
                                 color=color, alpha=0.8, linewidth=1.5,
                                 label=f'{sp}: {final_conc[i]:.2e}')

        # Fitted 결과
        fitted_positive_full = np.where(fitted_full > 0, fitted_full, np.nan)
        ax3.semilogy(wavelengths_full, fitted_positive_full, 'r-', 
                     linewidth=2.5, alpha=0.9, label=f'Fitted (R²={result["final_r2"]:.4f})', zorder=10)

        ax3.set_xlabel('Wavelength (nm)', fontsize=12)
        ax3.set_ylabel('Absorbance (log scale)', fontsize=12)
        ax3.set_title(f'Complete Spectral Analysis (Log Scale): {os.path.basename(filepath)}\n'
                      f'Measured (dots) + Fitted (red line) + Individual Species Contributions',
                      fontsize=14, fontweight='bold')
        
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([wavelengths_full.min(), wavelengths_full.max()])
        ax3.set_ylim([1e-6, 20]) # y축 고정

        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        output_file2 = os.path.join(output_dir,
                                    f'single_analysis_combined_log_{humidity_str}_{timestamp}.png')
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ 종합 그래프 (로그) 저장: {output_file2}")

        # ===============================================
        # CSV 파일 저장: 종합 플롯 데이터
        # ===============================================
        csv_data = {
            'Wavelength_nm': wavelengths_full,
            'Measured': absorbance_plot,
            'Fitted': fitted_full
        }
        
        # 각 화학종 기여도 추가
        for i, sp in enumerate(self.species_list):
            contribution = A_full[:, i] * final_conc[i]
            csv_data[f'{sp}_contribution'] = contribution
        
        # 농도 정보를 별도 행으로 추가하기 위해 먼저 데이터프레임 생성
        df_csv = pd.DataFrame(csv_data)
        
        # CSV 저장
        csv_file = os.path.join(output_dir,
                                f'single_analysis_combined_{humidity_str}_{timestamp}.csv')
        df_csv.to_csv(csv_file, index=False)
        
        # 농도 요약 정보를 별도 파일로 저장
        summary_data = {
            'Species': self.species_list,
            'Concentration': [final_conc[i] for i in range(len(self.species_list))],
            'Detection_Limit': [self.detection_limits.get(sp, 0) for sp in self.species_list],
            'DL_ppm': [self.detection_limits.get(sp, 0) / 2.46e10 for sp in self.species_list],
            'Unit': ['molecules/cm³'] * len(self.species_list)
        }
        df_summary = pd.DataFrame(summary_data)
        
        # 추가 정보
        summary_info = pd.DataFrame({
            'Species': ['_INFO_', '_INFO_', '_INFO_', '_INFO_', '_INFO_', '_INFO_'],
            'Concentration': [result['final_r2'], self.humidity_weight, 
                             result.get('n_saturated_excluded', 0),
                             self.saturation_threshold, self.path_length, 1e-3],
            'Detection_Limit': [0, 0, 0, 0, 0, 0],
            'DL_ppm': [0, 0, 0, 0, 0, 0],
            'Unit': ['R²', 'Humidity_Weight', 'Saturated_Excluded', 
                    'Saturation_Threshold', 'Path_Length_cm', 'DL_Absorbance']
        })
        df_summary = pd.concat([df_summary, summary_info], ignore_index=True)
        
        summary_file = os.path.join(output_dir,
                                    f'single_analysis_summary_{humidity_str}_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False)
        
        print(f"✓ 스펙트럼 데이터 저장: {csv_file}")
        print(f"✓ 농도 요약 저장: {summary_file}")

    def plot_time_series(self, df: pd.DataFrame, folder_path: str):
        """
        시간에 따른 농도 변화 그래프 [v8.8: 온도 서브플롯 추가, 3×2 → 4×2]
        """
        fig, axes = plt.subplots(4, 2, figsize=(14, 18))

        time_arr = df['Time'].values

        # ── 1. 모든 종 농도 변화 (로그) ──────────────────────────
        ax = axes[0, 0]
        for sp in self.species_list:
            conc = df[sp].values
            if np.any(conc > 0):
                if sp in ['HONO', 'HONO2']:
                    ax.semilogy(time_arr, conc, 'o--', label=f'{sp} (H)', alpha=0.5)
                else:
                    ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (log scale)')
        ax.set_title('All Species vs Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── 2. 주요 종 농도 변화 (선형) ──────────────────────────
        ax = axes[0, 1]
        main_species = ['O3', 'NO2', 'NO3']
        for sp in main_species:
            if sp in df.columns:
                ax.plot(time_arr, df[sp].values, 'o-', label=sp, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration')
        ax.set_title('Main Species (Linear Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 3. HONO/HONO2 변화 (습도 영향) ───────────────────────
        ax = axes[1, 0]
        if self.humidity_weight > 0:
            for sp in ['HONO', 'HONO2']:
                if sp in df.columns:
                    conc = df[sp].values
                    if np.any(conc > 0):
                        ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Concentration (log scale)')
            ax.set_title(f'HONO/HONO2 (Humidity: {self.humidity_weight*100:.0f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'HONO/HONO2 제외\n(습도 0%)',
                    ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        # ── 4. NOx 종 변화 ────────────────────────────────────────
        ax = axes[1, 1]
        nox_species = ['NO', 'NO2', 'N2O4']
        for sp in nox_species:
            if sp in df.columns:
                conc = df[sp].values
                if np.any(conc > 0):
                    ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration (log scale)')
        ax.set_title('NOx Species')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 5. [v8.8] 온도 시계열 ────────────────────────────────
        ax = axes[2, 0]
        if 'Temperature_C' in df.columns:
            temp_C = df['Temperature_C'].values
            ax.plot(time_arr, temp_C, 'ro-', alpha=0.8, linewidth=1.5, label='Temperature')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Temperature vs Time [v8.8]')
            ax.grid(True, alpha=0.3)
            # 온도 범위 주석
            ax.text(0.02, 0.97,
                    f'Min: {np.min(temp_C):.1f} °C\nMax: {np.max(temp_C):.1f} °C\n'
                    f'ΔT: {np.max(temp_C)-np.min(temp_C):.1f} K',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            # 프로파일 출처 표시
            src = self.temperature_profile['source'] if self.temperature_profile else '고정값'
            ax.text(0.98, 0.02, f'Source: {src}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                    color='gray')
        else:
            ax.text(0.5, 0.5, f'고정 온도\n{self.temperature-273.15:.1f} °C',
                    ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        # ── 6. [v8.8/v8.9] N2O4 / N2O5 평형상수 Keq 시계열 ─────
        ax = axes[2, 1]
        if 'Temperature_K' in df.columns:
            temps_K = df['Temperature_K'].values
            Keq_n2o4 = np.array([self.calculate_n2o4_equilibrium_constant(T) for T in temps_K])
            Keq_n2o5 = np.array([self.calculate_n2o5_equilibrium_constant(T) for T in temps_K])
            ax.semilogy(time_arr, Keq_n2o4, 'bs-', alpha=0.8, linewidth=1.5, label='Keq N₂O₄')
            ax.semilogy(time_arr, Keq_n2o5, 'r^-', alpha=0.8, linewidth=1.5, label='Keq N₂O₅')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Keq (cm³/molecule)')
            ax.set_title('Equilibrium Constants vs Time [v8.9]')
            ax.grid(True, alpha=0.3)
            ax.legend()
            # NO2/N2O4/N2O5 농도 오버레이 (우측 y축)
            ax2 = ax.twinx()
            for sp, fmt in [('NO2', 'g^--'), ('N2O4', 'm^--'), ('N2O5', 'c^--')]:
                if sp in df.columns and np.any(df[sp].values > 0):
                    ax2.semilogy(time_arr, df[sp].values, fmt, alpha=0.4, label=sp, linewidth=1)
            ax2.set_ylabel('Concentration (cm⁻³)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.legend(loc='lower right', fontsize=8)
        else:
            Keq_n2o4 = self.calculate_n2o4_equilibrium_constant(self.temperature)
            Keq_n2o5 = self.calculate_n2o5_equilibrium_constant(self.temperature)
            ax.text(0.5, 0.5,
                    f'고정 온도 ({self.temperature-273.15:.0f}°C)\n'
                    f'Keq N₂O₄: {Keq_n2o4:.2e}\nKeq N₂O₅: {Keq_n2o5:.2e}',
                    ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # ── 7. R² 시간 변화 ───────────────────────────────────────
        ax = axes[3, 0]
        ax.plot(time_arr, df['R2'].values, 'ko-', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R²')
        ax.set_title('Fitting Quality vs Time')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # ── 8. 플라즈마 타입 분포 ────────────────────────────────
        ax = axes[3, 1]
        plasma_counts = df['Plasma_Type'].value_counts()
        colors = {'O3_dominant': 'blue', 'NOx_dominant': 'red', 'mixed': 'purple'}
        bar_colors = [colors.get(pt, 'gray') for pt in plasma_counts.index]
        ax.bar(range(len(plasma_counts)), plasma_counts.values, color=bar_colors)
        ax.set_xticks(range(len(plasma_counts)))
        ax.set_xticklabels(plasma_counts.index, rotation=45)
        ax.set_ylabel('Count')
        ax.set_title('Plasma Type Distribution')

        info_text = f'Humidity: {self.humidity_weight*100:.0f}%'
        if 'Saturated_Excluded' in df.columns:
            total_excluded = df['Saturated_Excluded'].sum()
            if total_excluded > 0:
                info_text += f'\nSat. excluded: {total_excluded} pts'
        # [v8.8] 온도 정보 추가
        if 'Temperature_C' in df.columns:
            t_min = df['Temperature_C'].min()
            t_max = df['Temperature_C'].max()
            info_text += f'\nT: {t_min:.1f}~{t_max:.1f} °C'

        ax.text(0.98, 0.98, info_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 온도 프로파일 여부를 타이틀에 표기
        temp_mode = '동적 온도' if self.temperature_profile else f'고정 {self.temperature-273.15:.0f}°C'
        plt.suptitle(
            f'Batch Analysis Results v8.8 — {temp_mode}  |  Humidity: {self.humidity_weight*100:.0f}%',
            fontsize=13, fontweight='bold')
        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        output_file = os.path.join(folder_path, f'batch_plot_v88_{humidity_str}_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ 그래프 저장: {output_file}")

# ==========================================
# 메인 실행
# ==========================================

if __name__ == "__main__":

    # Cross section 경로 (고정)
    CROSS_SECTION_PATH = r"C:\Users\JY\Desktop\OAS code\cross-section"

    # === 피팅 옵션 (v8.3 기본값) ===
    correct_overfit = True
    correct_n2o4_equilibrium = True  # N2O4 평형 상한 제한 활성화
    temperature = 295.0  # 온도 (K) - 22°C
    noise_threshold = 1e-4  # Absorbance 노이즈 임계값
    
    # === NO3 피크 영역 강제 제약 (v8.4) ===
    no3_peak_constraint = True  # 피크 영역 강제 제약 활성화 (R² 감소 허용)
    no3_peak_range = (615, 635)  # NO3 피크 영역 (nm)
    no3_damping_factor = 0.95  # NO3 감소 비율 (0.90=빠름, 0.95=권장, 0.98=느림)
    no3_min_threshold = 0.01  # NO3 최소 임계값 (초기값 대비, 0.001=공격적, 0.005=권장, 0.01=보수적)
    no3_max_iterations = 150  # 최대 iteration (50=빠름, 150=권장, 300=완전제거)
    no3_window_nm = 3.0       # [v8.9] 경향 비교 윈도우 크기 (nm), 권장: 2.0~5.0
    
    # === UV 피팅 시작점 (v8.5) ===
    # 200nm 초반 영역 피팅 개선을 위한 옵션
    uv_start = 200.0  # UV 피팅 시작점 (nm) - 200nm부터 시작 권장
    
    # === N2O5 스케일 조절 (v8.7) ===
    # 200nm 초반에서 fitted > measured 문제 해결
    # - 0.0: N2O5 제외 - 200nm 초과 최소화
    # - 0.25: N2O5 25%로 제한
    # - 1.0: 제한 없음 (기존 방식) ← 기본값
    n2o5_scale = 1.0  # [v8.16] 자유 피팅 (200nm 초반 N2O5 크로스섹션이 O3보다 강함)
    apply_n2o5_stoich_cap = False  # [v8.16] False=NNLS값 사용(권장), True=min([NO2],[NO3]) 상한
    
    # === NO2 Vis Overfit 제약 (v8.14) ===
    no2_vis_constraint = True      # 350-450nm NO2 overfit 제약 (권장: True)
    no2_vis_range = (350, 450)     # 제약 구간 (nm)
    no2_vis_window_nm = 10.0       # 슬라이딩 윈도우 크기 (nm)

    # === HONO 상한 인자 (v8.14) ===
    # HONO_max = [NO2] × hono_no2_factor × (RH/100)
    # Stutz et al. (2004): k = 0.003~0.05 (환경·표면 의존)
    # 0.003: 하한(보수적), 0.01: 기본값, 0.05: 상한(습한 표면)
    hono_no2_factor = 0.01
    
    # === 조절 가이드 ===
    # 1. NO3가 너무 많이 감소하는 경우:
    #    - no3_damping_factor를 0.98로 증가 (천천히 감소)
    #    - no3_min_threshold를 0.01로 증가 (더 높은 최소값)
    #
    # 2. NO3가 충분히 감소하지 않는 경우:
    #    - no3_damping_factor를 0.90으로 감소 (빠르게 감소)
    #    - no3_max_iterations를 300으로 증가
    #
    # 3. 피크 영역을 조절하려면:
    #    - no3_peak_range를 변경 (예: (615, 640))
    #
    # 4. 200nm 초반 피팅:
    #    - n2o5_scale=1.0 권장 (N2O5 크로스섹션이 200nm 초반에서 O3보다 최대 29배 강함)
    #    - N2O5가 과도하게 나오는 경우: apply_n2o5_stoich_cap=True 또는 n2o5_scale 감소

    # 배치 분석기 초기화
    analyzer = BatchPlasmaAnalyzer(
        cross_section_path=CROSS_SECTION_PATH,
        path_length=5.0,
        correct_overfit=correct_overfit,
        correct_n2o4_equilibrium=correct_n2o4_equilibrium,
        temperature=temperature,
        noise_threshold=noise_threshold,
        no3_peak_constraint=no3_peak_constraint,
        no3_peak_range=no3_peak_range,
        no3_damping_factor=no3_damping_factor,
        no3_min_threshold=no3_min_threshold,
        no3_max_iterations=no3_max_iterations,
        no3_window_nm=no3_window_nm,
        uv_start=uv_start,
        n2o5_scale=n2o5_scale,
        apply_n2o5_stoich_cap=apply_n2o5_stoich_cap,
        no2_vis_constraint=no2_vis_constraint,
        no2_vis_range=no2_vis_range,
        no2_vis_window_nm=no2_vis_window_nm,
        hono_no2_factor=hono_no2_factor
    )

    # Cross sections 로드 (8종)
    analyzer.load_cross_sections()

    # 습도 설정
    analyzer.get_humidity_weight()

    # 분석 모드 선택 (단일/배치)
    mode = analyzer.get_analysis_mode()

    # 폴더 경로 입력
    folder_path = analyzer.get_folder_path()

    if mode == 'single':
        # ===== 단일 파일 분석 =====
        file_path = analyzer.get_single_file_path(folder_path)

        print("\n단일 파일 상세 분석 시작...")
        start_time = time.time()

        # 상세 분석 수행
        result = analyzer.analyze_single_file_detailed(file_path)

        if result:
            # 피팅 단계별 그래프 그리기
            analyzer.plot_single_file_results(result, file_path)

            print("\n" + "="*60)
            print("단일 파일 분석 완료!")
            print(f"최종 R²: {result['final_r2']:.4f}")
            print(f"포화 영역 제외: {result.get('n_saturated_excluded', 0)}개 점")
            print(f"Overfitting 보정: {'ON' if analyzer.correct_overfit else 'OFF'}")
            print(f"N2O4 평형 보정: {'ON' if analyzer.correct_n2o4_equilibrium else 'OFF'}")
            print(f"처리 시간: {time.time() - start_time:.1f}초")
            print("="*60)
        else:
            print("\n⚠ 분석 실패")

    else:
        # ===== 배치 분석 =====
        # [v8.8] 온도 프로파일 설정 (배치 전 인터랙티브 입력)
        analyzer.get_temperature_profile()

        # 개별 파일 플롯 옵션 선택
        plot_each = analyzer.get_batch_plot_option()
        
        start_time = time.time()
        analyzer.batch_analyze(folder_path, plot_each=plot_each)
        elapsed_time = time.time() - start_time

        # 결과 저장
        df_results = analyzer.save_batch_results(folder_path)

        if df_results is not None and len(df_results) > 0:
            # 시간 시리즈 플롯
            analyzer.plot_time_series(df_results, folder_path)

            print(f"\n총 처리 시간: {elapsed_time:.1f}초")
            print(f"파일당 평균: {elapsed_time/len(df_results):.1f}초")

        print("\n" + "="*60)
        print("배치 분석 완료!")
        print(f"습도 설정: {analyzer.humidity_weight*100:.0f}%")
        print(f"개별 파일 플롯: {'ON' if plot_each else 'OFF'}")
        print(f"Overfitting 보정: {'ON' if analyzer.correct_overfit else 'OFF'}")
        print(f"N2O4 평형 보정: {'ON' if analyzer.correct_n2o4_equilibrium else 'OFF'}")
        if analyzer.correct_n2o4_equilibrium:
            print(f"  온도: {analyzer.temperature:.1f} K")
        print(f"포화 threshold: {analyzer.saturation_threshold}")
        print("="*60)
