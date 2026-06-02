import cv2
import numpy as np
import urllib.request
import os

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. 원본 이미지
if not os.path.exists("images/original.png"):
    url = "https://raw.githubusercontent.com/cszn/FFDNet/master/testsets/Set12/08.png"
    urllib.request.urlretrieve(url, "images/original.png")
original = cv2.imread("images/original.png", 0)
original = cv2.resize(original, (512, 512))
cv2.imwrite("images/original.png", original)
print("원본 이미지 준비 완료")

# 2. PSF 생성 (15x15 Gaussian)
ksize, sigma = 21, 4.0
k1d = cv2.getGaussianKernel(ksize, sigma)
psf = (k1d @ k1d.T).astype(np.float64)
psf /= psf.sum()

# # 2. PSF 생성 (모션 블러일 경우)
# def make_motion_psf(length=25):
#     """수평 선형 모션 블러 PSF"""
#     psf = np.zeros((length, length), dtype=np.float64)
#     center = length // 2
#     psf[center, :] = 1.0
#     psf /= psf.sum()
#     return psf

# ksize = 25
# psf = make_motion_psf(length=ksize)

# 3. PSF → 512x512 패딩 후 ifftshift (FFT 위상 정렬)
def psf_to_fft(psf, size=512):
    """PSF를 이미지 중앙에 먼저 배치한 후 ifftshift"""
    pad = np.zeros((size, size), dtype=np.float64)
    
    h, w = psf.shape
    # 중앙에 PSF 배치
    cy, cx = size // 2, size // 2
    pad[cy - h//2 : cy + h//2 + 1,
        cx - w//2 : cx + w//2 + 1] = psf
    
    # 중앙 → (0,0) 이동
    pad = np.fft.ifftshift(pad)
    return pad

H_pad = psf_to_fft(psf)
H = np.fft.fft2(H_pad)

# 4. Degradation (FFT 기반 블러 + 노이즈)
orig_f = original.astype(np.float64) / 255.0
G_clean = np.fft.fft2(orig_f)

# FFT 도메인에서 블러 적용
G_blurred = G_clean * H
blurred = np.real(np.fft.ifft2(G_blurred))

# 노이즈 추가
np.random.seed(42)
noise_sigma = 3.0 / 255.0
noise = np.random.normal(0, noise_sigma, blurred.shape)
degraded_f = blurred + noise
degraded = np.clip(degraded_f * 255, 0, 255).astype(np.uint8)
cv2.imwrite("images/deg.png", degraded)

psnr_deg = cv2.PSNR(original, degraded)
print(f"블러 이미지 PSNR: {psnr_deg:.2f} dB  ← 기준선")

# PSF 시각화 저장
psf_vis = np.zeros((512, 512), dtype=np.uint8)
psf_vis[:ksize, :ksize] = (psf / psf.max() * 255).astype(np.uint8)
cv2.imwrite("images/ker.png", psf_vis)

# 5. Wiener Filter
def wiener_filter(deg_img, H, K):
    G = np.fft.fft2(deg_img.astype(np.float64) / 255.0)
    H_conj = np.conj(H)
    H_abssq = np.abs(H) ** 2
    F_hat = (H_conj / (H_abssq + K)) * G
    restored = np.real(np.fft.ifft2(F_hat))
    return np.clip(restored * 255, 0, 255).astype(np.uint8)

# 6. K값 정밀 탐색
print("\nK값 정밀 탐색")
K_values = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 
            0.006, 0.007, 0.008, 0.009, 0.01]
# # K_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
# K_values = [0.012, 0.015, 0.017, 0.020, 0.022, 0.025]
best_K, best_psnr, best_img = 0, 0, None
for K in K_values:
    r = wiener_filter(degraded, H, K)
    p = cv2.PSNR(original, r)
    cv2.imwrite(f"results/wiener_K{K}.png", r)
    if p > best_psnr:
        best_psnr, best_K, best_img = p, K, r
    print(f"  K={K:.4f} → PSNR: {p:.2f} dB")

cv2.imwrite("results/wiener_restored.png", best_img)

# 추가: SSIM도 계산
from skimage.metrics import structural_similarity as ssim
orig_arr = original
best_arr = best_img
ssim_blurred = ssim(original, degraded)
ssim_wiener  = ssim(original, best_img)

print(f"\nPhase 1 결과 요약")
print(f"  {'':12} {'PSNR':>8} {'SSIM':>6}")
print(f"  {'블러(기준선)':12} {psnr_deg:>8.2f} {ssim_blurred:>6.4f}")
print(f"  {'Wiener 복원':12} {best_psnr:>8.2f} {ssim_wiener:>6.4f}")
print(f"  최적 K값: {best_K}")
