import numpy as np
import cv2


def notch_filter(img,threshold=0.05):
    height, width = img.shape
    # 调用FFT得到频域
    fourier_trans = np.fft.fft2(img)
    # 中心化
    fourier_shift = np.fft.fftshift(fourier_trans)
    filter_1 = np.ones((height,width),dtype=np.float32)
    filter_2 = np.zeros((height,width),dtype=np.float32)
    center_frequency = (height//2,width//2)
    for i in range(height):
        for j in range(width):
            d = np.sqrt((i-center_frequency[0])**2 )
            # 去除不相干的频率
            if d > threshold*height:
                filter_1[i, j] = 0
                filter_2[i, j] = 1

    # 卷积后傅里叶逆变换
    fourier_inversed = np.fft.ifftshift(fourier_shift * filter_1)
    fourier_inversed_noise = np.fft.ifftshift(fourier_shift * filter_2)
    img_filtered = np.real(np.fft.ifft2(fourier_inversed))
    img_noise = np.real(np.fft.ifft2(fourier_inversed_noise))
    cv2.imwrite('./resources_labwork3/saturn_noise.png', img_noise)
    return img_filtered


saturn = cv2.imread('./resources_labwork3/saturn.png', cv2.IMREAD_GRAYSCALE)
saturn_filtered = notch_filter(saturn)
cv2.imwrite('./resources_labwork3/saturn_filtered.png', saturn_filtered)

