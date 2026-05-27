"""
NNLS 기반 플라즈마 화학종 농도 배치 분석 시스템 - 개선된 버전 v7
- [v7] 2단계 피팅 (확장): UV→O3/N2O5, Visible(350-650nm)→NO2/N2O4/NO3
- NO2 강제 피팅 개선
- N2O4 농도-NO2 평형에 의해 제한
- HONO/HONO2 후순위 피팅 (non-dry)
- 각 화학종 기여도 시각화
- NO2, HONO 강제 선택 가능
- 포화 영역 제외 (초기 설정: absorbance >= 2.75) 조절 가능
- single analysis plot - 종합 plot - 이미지와 .csv 파일로 저장
- batch analysis plot - 각 파일별 plot 확인 가능
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
    플라즈마 배치 분석 시스템 (습도 조절 가능)
    """

    # === MODIFIED: 파일명 시간 토큰 정규식 (공백 허용, 어디에 있어도 OK) ===
    TIME_TOKEN = re.compile(r'__\s*(\d+)\s*__')

    def __init__(self, cross_section_path: str, path_length: float = 5.0,
                 force_no2: bool = True, correct_overfit: bool = True,
                 correct_n2o4_equilibrium: bool = True, temperature: float = 295.0):
        """
        초기화

        Parameters:
        -----------
        cross_section_path : str
            Cross section 데이터 경로
        path_length : float
            흡광 경로 길이 (cm)
        force_no2 : bool
            NO2 강제 추가 활성화 (Step 3)
            - True: NO2 신호가 있는데 농도가 낮으면 강제 추가
            - False: 측정된 데이터만으로 피팅 (순수 NNLS)
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
        """
        self.cross_section_path = cross_section_path
        self.path_length = path_length

        # === 피팅 옵션 ===
        self.force_no2 = force_no2
        self.correct_overfit = correct_overfit
        self.correct_n2o4_equilibrium = correct_n2o4_equilibrium
        self.temperature = temperature

        # 화학종 순서 (HONO, HONO2 포함)
        self.species_order = ['O3', 'NO', 'NO2', 'NO3', 'N2O4', 'N2O5', 'HONO', 'HONO2']
        self.species_list = []
        self.cross_sections = {}
        self.fitting_range = (205, 650)  # nm
        
        # 습도 관련
        self.humidity_weight = None

        # === 추가: 포화 영역 threshold ===
        self.saturation_threshold = 2.75

        # 배치 처리 결과 저장
        self.batch_results = {}

        # 화학 반응 규칙 설정
        self.setup_chemical_rules()

        print("\n" + "="*60)
        print("BATCH PLASMA ANALYZER - 개선된 버전 v8")
        print("[v8] 2단계 피팅 (확장) + N2O4 평형 보정:")
        print("  1단계: UV(210-350nm) → O3, NO, N2O5")
        print("  2단계: Vis(350-650nm) → NO2, N2O4, NO3 (+HONO)")
        print(f"NO2 강제 피팅: {'ON' if self.force_no2 else 'OFF'}")
        print(f"Overfitting 보정: {'ON' if self.correct_overfit else 'OFF'}")
        print(f"N2O4 평형 보정: {'ON' if self.correct_n2o4_equilibrium else 'OFF'}")
        print(f"온도: {self.temperature:.1f} K")
        print(f"포화 영역 제외: absorbance >= {self.saturation_threshold}")
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
                'description': 'N2O4 + H2O -> HONO + HONO2'
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
                'rule': 'O3 > 2E14 → NO = 0 (동시 존재 불가)'
            },
            'NO2_O3': {
                'reactants': ['NO2', 'O3'],
                'products': ['NO3'],
                'rate': 'moderate',
                'description': 'NO2 + O3 -> NO3 + O2'
            }
        }

        print("\n[v8] 2단계 피팅 알고리즘 (확장) + N2O4 평형 보정:")
        print("  1단계: UV(210-350nm) → O3, NO, N2O5")
        print("  2단계: Vis(350-650nm) → NO2, N2O4, NO3 (+HONO)")
        print("  * O3-NO 규칙: O3 > 2E14 → NO 제거")
        print("  * HONO/HONO2: 습도 > 0일 때만 2단계에서 함께 피팅")
        print(f"  * 포화 영역 (absorbance >= {self.saturation_threshold}) 자동 제외")
        if self.correct_n2o4_equilibrium:
            print(f"  * N2O4 평형 보정: ON (T={self.temperature:.1f} K)")
    
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
        사용자로부터 습도 가중치 입력 받기 (0-1 범위)
        """
        print("\n" + "="*60)
        print("습도 설정")
        print("="*60)
        print("\n모든 파일에 동일한 습도 설정을 적용합니다.")
        print("HONO와 HONO2는 수분과의 반응으로 생성되는 2차 생성물입니다.")
        print("\nHONO 및 HONO2의 가중치를 0.0부터 1.0까지 선택해주세요:")
        print("  - 0.0: 완전 건조 (HONO/HONO2 제외)")
        print("  - 0.3: 약한 습도")
        print("  - 0.5: 중간 습도")
        print("  - 0.7: 높은 습도")
        print("  - 1.0: 매우 높은 습도 (HONO/HONO2 최대)")

        while True:
            try:
                humidity_input = input("\n습도 가중치 (0.0-1.0): ")
                humidity_value = float(humidity_input)

                if 0.0 <= humidity_value <= 1.0:
                    self.humidity_weight = humidity_value

                    if humidity_value == 0.0:
                        print(f"✓ 완전 건조 (HONO/HONO2 제외)")
                    elif humidity_value < 0.2:
                        print(f"✓ 매우 건조한 조건 (가중치: {self.humidity_weight:.2f})")
                    elif humidity_value < 0.4:
                        print(f"✓ 건조한 조건 (가중치: {self.humidity_weight:.2f})")
                    elif humidity_value < 0.6:
                        print(f"✓ 보통 습도 (가중치: {self.humidity_weight:.2f})")
                    elif humidity_value < 0.8:
                        print(f"✓ 습한 조건 (가중치: {self.humidity_weight:.2f})")
                    else:
                        print(f"✓ 매우 습한 조건 (가중치: {self.humidity_weight:.2f})")

                    print(f"\nHONO/HONO2 실제 피팅 가중치:")
                    print(f"  Step 1-3: {self.humidity_weight:.2f} × 0.3 = {self.humidity_weight*0.3:.3f}")
                    print(f"  Step 4 (overfitting 조정): {self.humidity_weight:.2f} × 0.3 × 0.2 = {self.humidity_weight*0.06:.3f}")
                    break
                else:
                    print("⚠ 0.0부터 1.0 사이의 값을 입력해주세요.")
            except ValueError:
                print("⚠ 올바른 숫자를 입력해주세요. (예: 0.5)")
            except KeyboardInterrupt:
                print("\n\n기본값 0.5 (중간 습도)으로 설정합니다.")
                self.humidity_weight = 0.5
                break

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

        # ===== 1단계: UV 영역 (210-350nm) - O3, NO, N2O5만 =====
        uv_region = (wavelengths >= 210) & (wavelengths <= 350)
        
        # 1단계에서 피팅할 종 (NO2, N2O4, NO3, HONO, HONO2 제외)
        stage1_indices = []
        for sp in ['O3', 'NO', 'N2O5']:
            if sp in self.species_list:
                stage1_indices.append(idx_map[sp])
        
        if np.any(uv_region) and len(stage1_indices) > 0:
            A_stage1 = A[uv_region][:, stage1_indices]
            b_stage1 = b[uv_region]
            
            conc_stage1, _ = nnls(A_stage1, b_stage1)
            
            for i, idx in enumerate(stage1_indices):
                conc[idx] = conc_stage1[i]
        
        fitted_stage1 = A @ conc
        r2_stage1 = 1 - np.sum((fitted_stage1 - b)**2) / np.sum((b - np.mean(b))**2)
        
        fitting_info['step1'] = {
            'concentrations': conc.copy(),
            'r2': r2_stage1,
            'description': 'Stage1: UV fitting (O3, NO, N2O5)'
        }

        # ===== O3-NO 경쟁 규칙 적용 =====
        o3_no_applied = False
        
        if o3_idx is not None and no_idx is not None:
            if conc[o3_idx] >= 2e14 and conc[no_idx] > 0:
                o3_no_applied = True
                
                stage1_no_NO = [idx for idx in stage1_indices if idx != no_idx]
                
                if len(stage1_no_NO) > 0:
                    A_stage1_no_NO = A[uv_region][:, stage1_no_NO]
                    conc_stage1_no_NO, _ = nnls(A_stage1_no_NO, b[uv_region])
                    
                    conc[no_idx] = 0
                    for i, idx in enumerate(stage1_no_NO):
                        conc[idx] = conc_stage1_no_NO[i]
        
        fitting_info['step1b'] = {
            'concentrations': conc.copy(),
            'r2': 1 - np.sum((A @ conc - b)**2) / np.sum((b - np.mean(b))**2),
            'description': f'O3-NO rule {"applied" if o3_no_applied else "not needed"}',
            'o3_no_applied': o3_no_applied
        }

        # ===== 2단계: Visible 영역 (350-650nm) - NO2, N2O4, NO3 (+HONO) =====
        vis_region = (wavelengths >= 350) & (wavelengths <= 650)
        
        if np.any(vis_region):
            fitted_so_far = A @ conc
            residual_vis = b[vis_region] - fitted_so_far[vis_region]
            residual_vis[residual_vis < 0] = 0
            
            if self.humidity_weight > 0:
                # 습도 있음: NO2 + N2O4 + NO3 + HONO + HONO2
                stage2_indices = []
                A_stage2_list = []
                
                if no2_idx is not None:
                    stage2_indices.append(no2_idx)
                    A_stage2_list.append(A[vis_region, no2_idx])
                
                if n2o4_idx is not None:
                    stage2_indices.append(n2o4_idx)
                    A_stage2_list.append(A[vis_region, n2o4_idx])
                
                if no3_idx is not None:
                    stage2_indices.append(no3_idx)
                    A_stage2_list.append(A[vis_region, no3_idx])
                
                if hono_idx is not None:
                    stage2_indices.append(hono_idx)
                    A_stage2_list.append(A[vis_region, hono_idx] * self.humidity_weight * 0.3)
                
                if hono2_idx is not None:
                    stage2_indices.append(hono2_idx)
                    A_stage2_list.append(A[vis_region, hono2_idx] * self.humidity_weight * 0.3)
                
                if len(A_stage2_list) > 0:
                    A_stage2 = np.column_stack(A_stage2_list)
                    conc_stage2, _ = nnls(A_stage2, residual_vis)
                    
                    for i, idx in enumerate(stage2_indices):
                        conc[idx] = conc_stage2[i]
            else:
                # 습도 0: NO2 + N2O4 + NO3만
                stage2_indices = []
                A_stage2_list = []
                
                if no2_idx is not None:
                    stage2_indices.append(no2_idx)
                    A_stage2_list.append(A[vis_region, no2_idx])
                
                if n2o4_idx is not None:
                    stage2_indices.append(n2o4_idx)
                    A_stage2_list.append(A[vis_region, n2o4_idx])
                
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
            'description': f'Stage2: Visible fitting (NO2, N2O4, NO3{" + HONO" if self.humidity_weight > 0 else ""})'
        }

        # ===== 3단계: Overfitting 보정 (습도>0일 때만) =====
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
                            if sp not in ['NO2', 'HONO', 'HONO2']:
                                other_contrib += A[no2_range, i] * conc[i]
                        
                        target = np.mean(np.maximum(measured_no2_region - other_contrib, 0))
                        
                        current_no2 = np.mean(A[no2_range, no2_idx] * conc[no2_idx])
                        current_hono = 0
                        if hono_idx is not None:
                            current_hono = np.mean(A[no2_range, hono_idx] * conc[hono_idx])
                        
                        current_total = current_no2 + current_hono
                        
                        if current_total > 0:
                            scale = min(target / current_total, 1.0)
                            
                            conc[no2_idx] *= scale
                            if hono_idx is not None:
                                conc[hono_idx] *= scale
                            if hono2_idx is not None:
                                conc[hono2_idx] *= scale
                            
                            fitted_stage3 = A @ conc
                            new_ratio = np.mean(fitted_stage3[no2_range]) / measured_mean
                            
                            fitting_info['step3'] = {
                                'concentrations': conc.copy(),
                                'r2': 1 - np.sum((fitted_stage3 - b)**2) / np.sum((b - np.mean(b))**2),
                                'description': f'Stage3: Overfit corrected ({overfitting_ratio:.1f}x -> {new_ratio:.1f}x)',
                                'hono_adjusted': True
                            }

        # ===== 3단계 추가: 500-650nm under-fitting 보정 =====
        red_region = (wavelengths >= 500) & (wavelengths <= 650)
        
        if np.any(red_region) and no3_idx is not None:
            fitted_current = A @ conc
            meas_red = np.mean(b[red_region])
            fit_red = np.mean(fitted_current[red_region])
            
            if meas_red > 0.001 and fit_red > 0:
                ratio_red = fit_red / meas_red
                if ratio_red < 0.95:  # under-fitting 감지
                    # NO3 농도 추가 조정
                    shortfall = meas_red - fit_red
                    no3_contrib_per_unit = np.mean(A[red_region, no3_idx])
                    if no3_contrib_per_unit > 0:
                        additional_no3 = shortfall / no3_contrib_per_unit
                        conc[no3_idx] += additional_no3
                        
                        fitted_after_red_correction = A @ conc
                        new_ratio_red = np.mean(fitted_after_red_correction[red_region]) / meas_red
                        
                        fitting_info['step3_red'] = {
                            'concentrations': conc.copy(),
                            'r2': 1 - np.sum((fitted_after_red_correction - b)**2) / np.sum((b - np.mean(b))**2),
                            'description': f'Stage3-Red: NO3 adjusted ({ratio_red:.2f}x -> {new_ratio_red:.2f}x)',
                            'no3_adjusted': True
                        }

        # ===== N2O4 평형 보정 (Step 4) =====
        if self.correct_n2o4_equilibrium:
            conc, fitting_info = self.correct_n2o4_by_equilibrium(conc, fitting_info)
            
            if 'n2o4_equilibrium' in fitting_info and fitting_info['n2o4_equilibrium'].get('correction_applied', False):
                fitted_after_n2o4 = A @ conc
                r2_after_n2o4 = 1 - np.sum((fitted_after_n2o4 - b)**2) / np.sum((b - np.mean(b))**2)
                
                fitting_info['step4_n2o4'] = {
                    'concentrations': conc.copy(),
                    'r2': r2_after_n2o4,
                    'description': fitting_info['n2o4_equilibrium']['description'],
                    'n2o4_corrected': True
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
        valid_mask = absorbance_full < self.saturation_threshold
        n_excluded = np.sum(~valid_mask)
        
        if n_excluded > 0:
            excluded_wl_min = wavelengths_full[~valid_mask].min()
            excluded_wl_max = wavelengths_full[~valid_mask].max()
            print(f"⚠ 포화 영역 제외: {n_excluded}개 점 (absorbance >= {self.saturation_threshold})")
            print(f"   제외 파장 범위: {excluded_wl_min:.1f} - {excluded_wl_max:.1f} nm")
        
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
            if step_name == 'n2o4_equilibrium':
                # 이 항목은 step4_n2o4에서 처리됨
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

        # 결과 출력
        print(f"\n최종 피팅 결과:")
        print(f"  R² = {final_r2:.4f}")
        for i, sp in enumerate(self.species_list):
            if final_conc[i] > 1e-10:
                print(f"  {sp}: {final_conc[i]:.3e}")

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
            fitted = A @ step_info['concentrations']
            fitting_stages[step_name] = {
                'concentrations': step_info['concentrations'],
                'fitted': fitted,
                'r2': step_info['r2'],
                'description': step_info['description']
            }

        # 전체 파장에 대한 피팅 결과 (그래프용)
        fitting_stages_full = {}
        for step_name, step_info in fitting_info.items():
            fitted_full = A_full @ step_info['concentrations']
            fitting_stages_full[step_name] = {
                'concentrations': step_info['concentrations'],
                'fitted': fitted_full,
                'r2': step_info['r2'],
                'description': step_info['description']
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
        """
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
        valid_mask = absorbance_full < self.saturation_threshold
        n_excluded = np.sum(~valid_mask)
        if n_excluded > 0:
            print(f"  ⚠ 포화 영역 제외: {n_excluded}개 점 (absorbance >= {self.saturation_threshold})")

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
            print(f"\n[{i}/{len(file_time_pairs)}] 분석 중: {filename}")
            print(f"  시간: {time_point}")

            result = self.analyze_single_file(filepath, time_point)

            if result:
                results_list.append(result)
                print(f"  R² = {result['r2']:.4f}")
                print(f"  플라즈마 타입: {result['plasma_type']}")

                # 주요 농도 출력
                conc = result['concentrations']
                for j, sp in enumerate(self.species_list):
                    if conc[j] > 1e-10:
                        print(f"    {sp}: {conc[j]:.2e}")
                
                # N2O4 평형 보정 정보 출력
                n2o4_eq_info = result.get('n2o4_equilibrium_info', {})
                if n2o4_eq_info.get('correction_applied', False):
                    print(f"    → N2O4 평형 보정: {n2o4_eq_info.get('ratio_before', 0):.1f}x → 1.0x")

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

        for result in self.batch_results:
            time_points.append(result['time'])
            r2_values.append(result['r2'])
            plasma_types.append(result['plasma_type'])
            saturated_excluded.append(result.get('n_saturated_excluded', 0))  # 추가

            for i, sp in enumerate(self.species_list):
                species_data[sp].append(result['concentrations'][i])

        # 데이터프레임 생성
        df = pd.DataFrame({
            'Time': time_points,
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

        # === 불연속 파장 영역 처리를 위한 헬퍼 함수 ===
        def insert_nan_at_gaps(wl, data, gap_threshold=2.0):
            """
            파장 간격이 threshold 이상인 곳에 NaN 삽입하여 선 끊기
            """
            if len(wl) < 2:
                return wl, data
            
            wl_diff = np.diff(wl)
            avg_diff = np.median(wl_diff)
            gap_indices = np.where(wl_diff > avg_diff * gap_threshold)[0]
            
            if len(gap_indices) == 0:
                return wl, data
            
            new_wl = list(wl)
            new_data = list(data)
            
            for offset, idx in enumerate(gap_indices):
                insert_pos = idx + 1 + offset
                new_wl.insert(insert_pos, (wl[idx] + wl[idx+1]) / 2)
                new_data.insert(insert_pos, np.nan)
            
            return np.array(new_wl), np.array(new_data)

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
            'Unit': ['molecules/cm³'] * len(self.species_list)
        }
        df_summary = pd.DataFrame(summary_data)
        
        # 추가 정보
        summary_info = pd.DataFrame({
            'Species': ['_INFO_', '_INFO_', '_INFO_', '_INFO_', '_INFO_'],
            'Concentration': [result['final_r2'], self.humidity_weight, 
                             result.get('n_saturated_excluded', 0),
                             self.saturation_threshold, self.path_length],
            'Unit': ['R²', 'Humidity_Weight', 'Saturated_Excluded', 
                    'Saturation_Threshold', 'Path_Length_cm']
        })
        df_summary = pd.concat([df_summary, summary_info], ignore_index=True)
        
        summary_file = os.path.join(output_dir,
                                    f'single_analysis_summary_{humidity_str}_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False)
        
        print(f"✓ 스펙트럼 데이터 저장: {csv_file}")
        print(f"✓ 농도 요약 저장: {summary_file}")

    def plot_time_series(self, df: pd.DataFrame, folder_path: str):
        """
        시간에 따른 농도 변화 그래프
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))

        time_arr = df['Time'].values

        # 1. 모든 종 농도 변화 (로그)
        ax = axes[0, 0]
        for sp in self.species_list:
            conc = df[sp].values
            if np.any(conc > 0):
                if sp in ['HONO', 'HONO2']:
                    ax.semilogy(time_arr, conc, 'o--', label=f'{sp} (H)', alpha=0.5)
                else:
                    ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration (log scale)')
        ax.set_title('All Species vs Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. 주요 종 농도 변화 (선형)
        ax = axes[0, 1]
        main_species = ['O3', 'NO2', 'NO3']

        for sp in main_species:
            if sp in df.columns:
                ax.plot(time_arr, df[sp].values, 'o-', label=sp, alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_title('Main Species (Linear Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. HONO/HONO2 변화 (습도 영향)
        ax = axes[1, 0]
        if self.humidity_weight > 0:
            for sp in ['HONO', 'HONO2']:
                if sp in df.columns:
                    conc = df[sp].values
                    if np.any(conc > 0):
                        ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)

            ax.set_xlabel('Time')
            ax.set_ylabel('Concentration (log scale)')
            ax.set_title(f'HONO/HONO2 (Humidity: {self.humidity_weight*100:.0f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'HONO/HONO2 제외\n(습도 0%)',
                    ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

        # 4. NOx 종 변화
        ax = axes[1, 1]
        nox_species = ['NO', 'NO2', 'N2O4']
        for sp in nox_species:
            if sp in df.columns:
                conc = df[sp].values
                if np.any(conc > 0):
                    ax.semilogy(time_arr, conc, 'o-', label=sp, alpha=0.7)

        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration (log scale)')
        ax.set_title('NOx Species')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. R² 시간 변화
        ax = axes[2, 0]
        ax.plot(time_arr, df['R2'].values, 'ko-', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('R²')
        ax.set_title('Fitting Quality vs Time')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # 6. 플라즈마 타입 분포 + 포화 제외 통계
        ax = axes[2, 1]
        plasma_counts = df['Plasma_Type'].value_counts()
        colors = {'O3_dominant': 'blue', 'NOx_dominant': 'red', 'mixed': 'purple'}
        bar_colors = [colors.get(pt, 'gray') for pt in plasma_counts.index]

        ax.bar(range(len(plasma_counts)), plasma_counts.values, color=bar_colors)
        ax.set_xticks(range(len(plasma_counts)))
        ax.set_xticklabels(plasma_counts.index, rotation=45)
        ax.set_ylabel('Count')
        ax.set_title('Plasma Type Distribution')

        # 습도 및 포화 정보 추가
        info_text = f'Humidity: {self.humidity_weight*100:.0f}%'
        if 'Saturated_Excluded' in df.columns:
            total_excluded = df['Saturated_Excluded'].sum()
            if total_excluded > 0:
                info_text += f'\nSat. excluded: {total_excluded} pts'

        ax.text(0.98, 0.98, info_text,
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.suptitle(f'Batch Analysis Results - Improved (Humidity: {self.humidity_weight*100:.0f}%)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        humidity_str = f"humid{int(self.humidity_weight*100)}"
        output_file = os.path.join(folder_path, f'batch_plot_improved_{humidity_str}_{timestamp}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✓ 그래프 저장: {output_file}")

# ==========================================
# 메인 실행
# ==========================================

if __name__ == "__main__":

    # Cross section 경로 (고정)
    CROSS_SECTION_PATH = r"G:\ASRock\Desktop\코드\20250917_활성종 분석 코드\NNLS_A.I\Cross_sections_out"

    # === 피팅 옵션 (v8 기본값) ===
    force_no2 = True
    correct_overfit = True
    correct_n2o4_equilibrium = True  # N2O4 평형 상한 제한 활성화
    temperature = 295.0  # 온도 (K) - 22°C

    # 배치 분석기 초기화
    analyzer = BatchPlasmaAnalyzer(
        cross_section_path=CROSS_SECTION_PATH,
        path_length=5.0,
        force_no2=force_no2,
        correct_overfit=correct_overfit,
        correct_n2o4_equilibrium=correct_n2o4_equilibrium,
        temperature=temperature
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
            print(f"NO2 강제 추가: {'ON' if analyzer.force_no2 else 'OFF'}")
            print(f"Overfitting 보정: {'ON' if analyzer.correct_overfit else 'OFF'}")
            print(f"N2O4 평형 보정: {'ON' if analyzer.correct_n2o4_equilibrium else 'OFF'}")
            print(f"처리 시간: {time.time() - start_time:.1f}초")
            print("="*60)
        else:
            print("\n⚠ 분석 실패")

    else:
        # ===== 배치 분석 =====
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
        print(f"NO2 강제 추가: {'ON' if analyzer.force_no2 else 'OFF'}")
        print(f"Overfitting 보정: {'ON' if analyzer.correct_overfit else 'OFF'}")
        print(f"N2O4 평형 보정: {'ON' if analyzer.correct_n2o4_equilibrium else 'OFF'}")
        if analyzer.correct_n2o4_equilibrium:
            print(f"  온도: {analyzer.temperature:.1f} K")
        print(f"포화 threshold: {analyzer.saturation_threshold}")
        print("="*60)
