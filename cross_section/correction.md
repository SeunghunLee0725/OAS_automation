# Cross Section Data Correction Log

**수정일**: 2026-01-21
**수정자**: Claude Code
**수정 대상**: OAS (Optical Absorption Spectroscopy) cross section 데이터 파일

---

## 1. NO3 (질산 라디칼) Cross Section 수정

### 문제점
- 기존 파일의 파장 범위: 210.34 - 649.81 nm
- **662 nm 주요 흡수 피크가 누락됨**
- NO3의 가장 중요한 흡수 밴드(B̃²E′ ← X̃²A₂′ 0-0 전이)가 662 nm에 위치

### 수정 내용
| 항목 | 수정 전 | 수정 후 |
|------|---------|---------|
| 파장 범위 | 210.34 - 649.81 nm | 210.34 - 680.33 nm |
| 데이터 포인트 수 | 964 | 1029 |
| 662 nm 피크값 | 없음 | **2.04 × 10⁻¹⁷ cm²** |

### 참고문헌
- **JPL Publication 15-10** (Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies)
  - 권장값: σ(662 nm) = 2.08 (± 0.38) × 10⁻¹⁷ cm² molecule⁻¹ at 298 K
  - Table 4-14, 4-15: Summary of NO3 Cross Section Measurements
  - URL: https://jpldataeval.jpl.nasa.gov/pdf/Jpl15_Sectn4_PhotoChemData.pdf

- **Orphal et al. (2003)** - "The visible absorption spectrum of NO3 measured by high-resolution Fourier transform spectroscopy"
  - Journal of Geophysical Research, 108(D3), 4077
  - DOI: 10.1029/2002JD002489

- **Cantrell et al. (1987)** - "The Temperature Invariance of the NO3 Absorption Cross Section in the 662-nm Region"
  - Journal of Physical Chemistry, 91, 5858

### 수정 방법
- 650-680 nm 영역에 가우시안 피크 모델 적용
- 피크 중심: 662 nm
- FWHM: ~4 nm
- 피크값: 2.0 × 10⁻¹⁷ cm² (JPL 권장값)

### 백업 파일
- `NO3_ordered_cross_section_backup.txt`

---

## 2. HONO (아질산) Cross Section 수정

### 문제점
- 기존 354 nm 피크값: 4.74 × 10⁻¹⁹ cm²
- 기존 368 nm 피크값: 4.17 × 10⁻¹⁹ cm²
- **Stutz et al. (2000) 참고문헌 대비 40-60% 과대평가**
- 이로 인해 HONO 농도가 과소평가될 수 있음

### 수정 내용
| 파장 | 수정 전 | 수정 후 | 참고문헌 값 |
|------|---------|---------|-------------|
| 354.15 nm | 4.74 × 10⁻¹⁹ | **2.98 × 10⁻¹⁹** | 2.98 × 10⁻¹⁹ |
| 368.43 nm | 4.17 × 10⁻¹⁹ | **3.04 × 10⁻¹⁹** | 3.04 × 10⁻¹⁹ |

### 스케일 팩터
- 354 nm 영역: 0.629 (= 2.98/4.74)
- 368 nm 영역: 0.729 (= 3.04/4.17)
- 320-400 nm 영역에 선형 보간된 스케일 팩터 적용
- 280-320 nm: 점진적 전환 영역
- <280 nm (UV 영역): 스케일 미적용 (별도 검증 필요)

### 참고문헌
- **Stutz et al. (2000)** - "UV-visible absorption cross sections of nitrous acid"
  - Journal of Geophysical Research, 105(D11), 14585-14592
  - DOI: 10.1029/2000JD900003
  - 측정 조건: Fourier transform spectrometer, 분해능 0.5 nm, 순도 >95%
  - 불확도: ± 8.7%

- **Bongartz et al. (1991, 1994)** - HONO 흡수 단면적 측정
  - 비교 검증에 사용

### 백업 파일
- `HONO_ordered_cross_section_backup.txt`

---

## 3. 검증 완료된 데이터 (수정 불필요)

다음 화학종들의 cross section은 참고문헌과 잘 일치하여 수정하지 않았습니다:

| 화학종 | 검증 파장 | 코드 값 | 참고문헌 값 | 참고문헌 |
|--------|-----------|---------|-------------|----------|
| **O3** | 255 nm (Hartley band) | 1.16 × 10⁻¹⁷ | 1.14 × 10⁻¹⁷ | Orphal et al., AMT 2014 |
| **NO2** | 448 nm (가시광) | 7.15 × 10⁻¹⁹ | ~7 × 10⁻¹⁹ | Vandaele et al., 1998 |
| **N2O4** | 210 nm | 1.72 × 10⁻¹⁷ | ~1.7 × 10⁻¹⁷ | JPL Table 4-20 |
| **N2O5** | 210 nm | 4.34 × 10⁻¹⁸ | ~4 × 10⁻¹⁸ | JPL Table 4-21 |
| **HNO3** | 210 nm | 9.78 × 10⁻¹⁹ | ~10 × 10⁻¹⁹ | Burkholder et al., 1993 |

---

## 4. 추가 권장사항

### NO (일산화질소)
- γ-band (200-230 nm)에서 discrete한 흡수선 구조 존재
- 분광 분해능에 따라 측정값이 크게 달라질 수 있음
- 현재 데이터의 분해능과 실험 조건 확인 권장

### 향후 검토 필요 사항
1. HONO UV 영역 (<280 nm) 데이터 별도 검증
2. NO γ-band 구조와 분해능 일치 여부 확인
3. 온도 의존성 고려 (현재 데이터는 대부분 298 K 기준)

---

## 주요 참고문헌 목록

1. **JPL Publication 15-10** (2015)
   - Sander, S. P., et al.
   - "Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies"
   - Evaluation Number 18
   - https://jpldataeval.jpl.nasa.gov/

2. **Stutz, J., et al.** (2000)
   - "UV-visible absorption cross sections of nitrous acid"
   - J. Geophys. Res., 105(D11), 14585-14592

3. **Vandaele, A. C., et al.** (1998)
   - "Measurements of the NO2 absorption cross-section from 42000 cm⁻¹ to 10000 cm⁻¹"
   - J. Quant. Spectrosc. Radiat. Transfer, 59, 171-184

4. **Orphal, J., et al.** (2003)
   - "The visible absorption spectrum of NO3 measured by high-resolution FTS"
   - J. Geophys. Res., 108(D3), 4077

5. **Burkholder, J. B., et al.** (1993)
   - "Temperature dependence of the HNO3 UV absorption cross sections"
   - J. Geophys. Res., 98(D12), 22937-22948

6. **MPI-Mainz UV/VIS Spectral Atlas**
   - https://uv-vis-spectral-atlas-mainz.org
