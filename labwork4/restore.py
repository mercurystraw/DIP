import cv2
import numpy as np
from labwork4.degeneration import turbulence, motion_blur, turbulence_add_noise, motion_blur_add_noise


# 逆滤波函数
def inverse_filter(degraded_image, H , G): # 全逆滤波效果很差，用带有半径的逆滤波
    # g = np.float32(degraded_image) / 255.0
    # G = np.fft.fft2(g)  # 转换到频域
    # G = np.fft.fftshift(G)  # 频谱中心化


    # rows, cols = H.shape
    # center_row, center_col = rows // 2, cols // 2
    # Y, X = np.ogrid[:rows, :cols]
    # distance_from_center = (X - center_col) ** 2 + (Y - center_row) ** 2
    # H_radius = np.ones((rows, cols), dtype=np.float32)
    #
    # r = 10
    # H_radius[distance_from_center <= r] = H[distance_from_center <= r].real
    # # 进行逆滤波
    # X = np.where(H_radius != 0, G * H_radius, 0)  # 遇到H为零的情况返回0
    X = G / H

    X = np.fft.ifftshift(X)  # 逆频谱中心化

    # 反变换
    restored_image = np.real(np.fft.ifft2(X))
    restored_image = np.clip(restored_image, 0, 1)
    restored_image = (restored_image * 255).astype(np.uint8)

    return restored_image


# 维纳滤波函数
def wiener_filter(degraded_image, original_image, H, noise):

    g = np.float32(degraded_image) / 255.0
    G = np.fft.fft2(g)  # 转换到频域
    G = np.fft.fftshift(G)  # 频谱中心化

    # 计算原本图画图像的功率谱
    f_original = np.float32(original_image) / 255.0
    F_original = np.fft.fft2(f_original)
    F_original = np.fft.fftshift(F_original)
    original_image_power = np.abs(F_original) ** 2

    # 有噪声/没噪声的情况
    if noise is not None:
        n = np.float32(noise) / 255.0
        N = np.fft.fft2(n)
        N = np.fft.fftshift(N)
        noise_power = np.abs(N) ** 2
    else:
        noise_power = 0.0

    H_power = np.abs(H) ** 2

    F_estimate = (H_power/(H*(H_power+noise_power/original_image_power)))*G

    # 反变换
    restored_image = np.real(np.fft.ifft2(F_estimate))
    restored_image = np.clip(restored_image, 0, 1)
    restored_image = (restored_image * 255).astype(np.uint8)

    return restored_image


# 原本的图像
countryside_image = cv2.imread("./resources_labwork4/countryside.png", cv2.IMREAD_GRAYSCALE)
digital_image = cv2.imread("./resources_labwork4/digital.png", cv2.IMREAD_GRAYSCALE)

# 读取退化后的图像并复原
countryside_degenerated = cv2.imread("./resources_labwork4/countryside_degenerated.png", cv2.IMREAD_GRAYSCALE)
countryside_degenerated_add_noise = cv2.imread("./resources_labwork4/countryside_degenerated_add_noise.png", cv2.IMREAD_GRAYSCALE)
digital_degenerated = cv2.imread("./resources_labwork4/digital_motionBlur.png", cv2.IMREAD_GRAYSCALE)
digital_degenerated_add_noise = cv2.imread("./resources_labwork4/digital_motionBlur_add_noise.png", cv2.IMREAD_GRAYSCALE)

# 获取之前退化函数使用的的H
_, H_turbulence, F_turbulence = turbulence(countryside_image, 0.001)
_, H_motionBlur, F_motionBlur = motion_blur(digital_image)
_, F_noise_turbulence = turbulence_add_noise(countryside_image)
_, F_noise_motionBlur = motion_blur_add_noise(digital_image)
# 计算添加的噪声
countryside_noise = countryside_degenerated_add_noise.astype(np.float32) - countryside_degenerated.astype(np.float32)
# 无噪声退化直接逆滤波
countryside_restored_inverse_filter = inverse_filter(countryside_degenerated, H_turbulence, F_turbulence)
# 有噪声逆滤波
countryside_addNoise_restored_inverse_filter = inverse_filter(countryside_degenerated_add_noise, H_turbulence, F_noise_turbulence)
# 有噪声维纳滤波
countryside_addNoise_restored_wiener_filter = wiener_filter(countryside_degenerated_add_noise, countryside_image, H_turbulence, countryside_noise)

# 保存恢复的countryside图像
cv2.imwrite("./resources_labwork4/countryside_restored_inverse_filter.png", countryside_restored_inverse_filter)
cv2.imwrite("./resources_labwork4/countryside_addNoise_restored_inverse_filter.png", countryside_addNoise_restored_inverse_filter)
cv2.imwrite("./resources_labwork4/countryside_addNoise_restored_wiener_filter.png", countryside_addNoise_restored_wiener_filter)

# 计算添加的噪声
digital_noise = digital_degenerated_add_noise.astype(np.float32) - digital_degenerated.astype(np.float32)
# 无噪声退化直接逆滤波,
digital_restored_inverse_filter = inverse_filter(digital_degenerated, H_motionBlur, F_motionBlur)
# 有噪声逆滤波
digital_addNoise_restored_inverse_filter = inverse_filter(digital_degenerated_add_noise, H_motionBlur, F_noise_motionBlur)
# 有噪声维纳滤波
digital_addNoise_restored_wiener_filter = wiener_filter(digital_degenerated_add_noise, digital_image, H_motionBlur, digital_noise)

# 保存恢复的digital图像
cv2.imwrite("./resources_labwork4/digital_restored_inverse_filter.png", digital_restored_inverse_filter)
cv2.imwrite("./resources_labwork4/digital_addNoise_restored_inverse_filter.png", digital_addNoise_restored_inverse_filter)
cv2.imwrite("./resources_labwork4/digital_addNoise_restored_wiener_filter.png", digital_addNoise_restored_wiener_filter)

