import cv2
import numpy as np
import pywt
# 读取TIFF格式的图像
image = cv2.imread('./resources_labwork6/sinePulses.tif', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声
mean = 0
sigma = 25
gaussian_noise = np.random.normal(mean, sigma, image.shape)
noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
cv2.imwrite('./resources_labwork6/noisy_image.tif', noisy_image)


# 小波去噪
def wavelet(image, wavelet, threshold=20):
    # 进行小波分解
    coeffs = pywt.wavedec2(image, wavelet)

    # 处理细节系数
    coeffs_thresholded = list(coeffs)
    for i in range(1, len(coeffs_thresholded)):
        coeffs_thresholded[i] = tuple(
            pywt.threshold(c, threshold, mode='soft') for c in coeffs_thresholded[i]
        )

    # 小波重构
    denoised_image = pywt.waverec2(coeffs_thresholded, wavelet)
    return np.clip(denoised_image, 0, 255).astype(np.uint8)


def wavelet_edge_detection(image, wavelet, level=3):
    # 将图像转换为浮点数类型
    image = image.astype(np.float32) / 255.0
    # 小波分解
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 提取近似分量和细节系数（水平、垂直、对角线）
    approx_coeff = coeffs[0]
    detail_coeffs = coeffs[1:]

    approx_coeff *= 0

    # 反变换
    reconstructed_image = pywt.waverec2([approx_coeff] + detail_coeffs, wavelet)
    edge_image = np.clip(reconstructed_image * 255, 0, 255).astype(np.uint8)
    return edge_image


# 不同的小波函数
wavelets = ['haar', 'db1', 'db2', 'sym2', 'coif1']
for wavelet_type in wavelets:
    denoised_image = wavelet(noisy_image, wavelet_type)
    cv2.imwrite('./resources_labwork6/denoised_image_' + wavelet_type + '.tif', denoised_image)

    edges = wavelet_edge_detection(denoised_image, wavelet_type)
    cv2.imwrite('./resources_labwork6/edges_' + wavelet_type + '.tif', edges)

print('图片处理完成！')


