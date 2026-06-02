#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _DEBUG
#pragma comment (lib,"opencv_world4120d")
#else
#pragma comment (lib,"opencv_world4120")
#endif

using namespace cv;
using namespace std;

// 스펙트럼(주파수 영역) 시각화 함수
void showSpectrum(const std::string& winName, const Mat& src, bool shuffle = true) {
    Mat mag;
    Mat chan[2];
    split(src, chan);
    magnitude(chan[0], chan[1], mag);

    Mat tmp(mag.size(), mag.type());
    if (shuffle) {
        int w = mag.cols / 2;
        int h = mag.rows / 2;
        // 사분면 교체 (DFT 결과를 중앙으로 집중)
        mag(Rect(0, 0, w, h)).copyTo(tmp(Rect(w, h, w, h)));
        mag(Rect(w, 0, w, h)).copyTo(tmp(Rect(0, h, w, h)));
        mag(Rect(0, h, w, h)).copyTo(tmp(Rect(w, 0, w, h)));
        mag(Rect(w, h, w, h)).copyTo(tmp(Rect(0, 0, w, h)));
    }
    else {
        mag.copyTo(tmp);
    }

    log(tmp + 1, tmp);
    imshow(winName, tmp / 10);
}

int main() {
    // 1. 이미지 로드 및 정규화
    Mat deg = imread("C:/01Jisoo/openCV/wiener-filter/deg.png", 0);
    if (deg.empty()) return -1;
    deg.convertTo(deg, CV_32F, 1 / 255.f);

    // 2. 커널(PSF) 로드 및 정규화
    Mat ker = imread("C:/01Jisoo/openCV/wiener-filter/ker.png", 0);
    if (ker.empty()) return -1;
    ker.convertTo(ker, CV_32F, 1 / 255.f);
    ker /= sum(ker)[0];

    // 3. DFT 수행
    Mat G, H;
    dft(deg, G, DFT_COMPLEX_OUTPUT);
    dft(ker, H, DFT_COMPLEX_OUTPUT);

    showSpectrum("PSF Spectrum", H * 100);

    // 4. 위너 필터링 계산
    Mat H_abssq;
    mulSpectrums(H, H, H_abssq, 0, true); // |H|^2 계산

    Mat F_hat(G.size(), CV_32FC2);
    float K = 0.0045f; // 노이즈 대비 신호 비율 매개변수

    for (int y = 0; y < F_hat.rows; y++) {
        for (int x = 0; x < F_hat.cols; x++) {
            Vec2f g = G.at<Vec2f>(y, x);
            Vec2f h = H.at<Vec2f>(y, x);
            float h_abssq = H_abssq.at<Vec2f>(y, x)[0];

            float d = h_abssq + K;

            if (d == 0) {
                F_hat.at<Vec2f>(y, x) = Vec2f(0.0f, 0.0f);
            }
            else {
                // 복소수 나눗셈: G * H* / (|H|^2 + K)
                // H의 켤레복소수(h*)를 활용한 분자 계산
                float n_real = g[0] * h[0] + g[1] * h[1];
                float n_imagi = g[1] * h[0] - g[0] * h[1];

                F_hat.at<Vec2f>(y, x)[0] = n_real / d;
                F_hat.at<Vec2f>(y, x)[1] = n_imagi / d;
            }
        }
    }

    // 5. 역 DFT 수행 및 결과 출력
    Mat restored_image;
    idft(F_hat, restored_image, DFT_REAL_OUTPUT | DFT_SCALE);

    imshow("Degraded image", deg);
    imshow("Well Restored Wiener Filtering image", restored_image);

    waitKey(0);
    return 0;
}