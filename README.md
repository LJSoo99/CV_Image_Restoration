# 주파수 도메인 영상 복원에서 딥러닝으로
**Image Restoration: From Wiener Filter to Deep Learning**

---

### Results

| 방법 | PSNR (dB) | SSIM | 특징 |
|---|---|---|---|
| 블러 (기준선) | 24.38 | 0.6380 | 입력 |
| Wiener Filter | 26.95 | 0.7225 | PSF·K값 필요 |
| **DPIR (DL)** | **27.52** | **0.7652** | 데이터 기반, PSF 불필요 |

<img width="2072" height="512" alt="comparison_4panel (1)" src="https://github.com/user-attachments/assets/2989b901-baaa-4e1f-9272-909096bbff13" />

---

## 1. 문제 정의

카메라 흔들림·초점 오류로 인한 영상 블러는 의료·감시·사진 분야에서 핵심적인 문제라고 생각합니다. 
블러 제거 없이는 후속 비전 처리(객체 탐지, 분류)의 정확도가 저하됩니다.

**기존 접근 — Wiener Filter:**  
PSF(Point Spread Function)를 알고 있을 때 주파수 도메인에서 역필터링을 수행합니다.

<img width="704" height="138" alt="534549581-3d15cb87-82cc-41b7-99e7-edf3cee4490f" src="https://github.com/user-attachments/assets/b3fdd1a6-8f07-4d7b-8a10-72eda2c56bcf" />

---

## 2. 구현 및 실험 환경

| 항목 | 내용 |
|---|---|
| 언어 | C++ (OpenCV 4.12), Python 3.12 |
| 딥러닝 모델 | DPIR / DRUNet (IEEE TPAMI 2021) |
| 실험 이미지 | Set12 benchmark (Lena, 512×512, grayscale) |
| 블러 유형 | Gaussian PSF (ksize=21, σ=4.0) + Gaussian noise (σ=3) |

---

## 3. Phase 1 — Wiener Filter 구현 및 한계 확인

### 3.1 복원 결과

| 방법 | PSNR (dB) | SSIM | 비고 |
|---|---|---|---|
| 블러 (기준선) | 24.38 | 0.6380 | 입력 이미지 |
| **Wiener Filter** | **26.95** | **0.7225** | K=0.004, +2.57 dB |

### 3.2 K값 민감도 실험 (Blind Deblurring 한계)

PSF를 알고 있어도 노이즈 파라미터 K를 잘못 설정하면 복원 품질이 급격히 저하됩니다.

| K값 | PSNR (dB) | 현상 |
|---|---|---|
| 0.0001 | 17.34 | 너무 작음 → 노이즈 증폭 |
| 0.001 | 27.21 | 과소 추정 |
| **0.004** | **26.95** | **최적** |
| 0.02 | 25.05 | 과대 추정 → 과도한 평활화 |
| 0.1 | 23.32 | 너무 큼 → 블러 잔존 |

**한계 확인:**

- **PSF 의존성:** PSF와 K값을 모두 정확히 알아야 최적 복원 가능
- **Blind Deblurring 불가:** 실제 환경에서 PSF를 정확히 알기 어려움
- **비균일 블러 처리 불가:** 초점 오류·렌즈 수차 등 공간 변이 블러에 적용 불가

---

## 4. Phase 2 — 딥러닝 기반 복원과 정량적 비교

### 4.1 접근 방법 선택 이유

**왜 DPIR인가:**  
Gaussian 블러 + 노이즈 복원에 특화된 Plug-and-Play 방식으로,  
PSF 없이 데이터 기반 denoiser(DRUNet)를 반복 적용하여 복원합니다.  
Half-Quadratic Splitting(HQS) 알고리즘으로 데이터 충실도와 prior를 교대 최적화합니다.

### 4.2 최종 비교표

| 방법 | PSNR (dB) | SSIM | 특징 |
|---|---|---|---|
| 블러 (기준선) | 24.38 | 0.6380 | 입력 |
| Wiener Filter | 26.95 | 0.7225 | PSF·K값 필요 |
| **DPIR (DL)** | **27.52** | **0.7652** | 데이터 기반, PSF 불필요 |

**Wiener 대비 DPIR: +0.57 dB / +0.0427 SSIM**

> "PSF를 정확히 알고 최적 K값을 사용한 Wiener Filter(26.95 dB)보다,  
> 데이터에서 학습한 DPIR(27.52 dB)이 더 높은 복원 품질을 달성했습니다.  
> 특히 SSIM에서 +0.0427의 개선은 구조적 디테일 보존 측면에서 의미 있는 차이라고 생각합니다."

---

## 5. Phase 3 — 한계 및 향후 연구 방향

### 5.1 현재 구현의 한계

- **한계 1:** 단일 이미지 실험 — 대규모 데이터셋에서 일반화 성능 미검증
- **한계 2:** 사전학습 모델 추론만 사용 — 도메인 특화 fine-tuning 미적용
- **한계 3:** 실시간 처리 미최적화 — HQS 반복(iter=24)으로 추론 속도 제한

### 5.2 향후 연구 방향

- **방향 1:** Diffusion Model 기반 복원 (DiffPIR) — 생성 모델을 prior로 활용
- **방향 2:** Blind Deblurring을 위한 PSF 자동 추정 모듈

---

## 6. 프로젝트 구조

```
├── wiener-filter/
│   └── wiener-filter.cpp   # C++ Wiener Filter 구현 (OpenCV DFT)
├── project/
│   └── phase1.py           # 실험 파이프라인 (이미지 생성 + Wiener)
│   └── phase2.py           # 딥러닝 파이프라인 (DPIR)
├── images/
│   ├── original.png              # 원본 (Set12 #08, Lena)
│   ├── deg.png                   # 블러+노이즈 이미지
│   └── ker.png                   # PSF 커널 시각화
├── results/
│   ├── wiener_restored.png       # Wiener 복원 결과
│   └── dpir_restored.png         # DPIR 복원 결과
└── README.md
```

---

## 7. 참고 문헌

- Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior" (IEEE TPAMI 2021) — DPIR
- Gonzalez & Woods, "Digital Image Processing" — Wiener Filter
- Set12 Benchmark Dataset (cszn/FFDNet)


### 3. 실험 환경 (Experimental Environment)

| 항목 (Item) | 내용 (Description) |
| :--- | :--- |
| **Set** | 12 Lena, $512 \times 512$ grayscale |
| **블러 (Blur)** | Gaussian PSF ($\text{ksize}=21$, $\sigma=4.0$) |
| **노이즈 (Noise)** | Gaussian noise ($\sigma=3$) |
| **구현 (Implementation)** | C++ OpenCV 4.12 / Python 3.12 + PyTorch |
