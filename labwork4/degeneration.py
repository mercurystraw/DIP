import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def turbulence(image, k =0.001): # k=0.0025剧烈湍流；k=0.001中等湍流；k=0.00025低湍流
    f = np.float32(image) / 255.0
    F = np.fft.fft2(f) # 转换到频域
    F = np.fft.fftshift(F) # 频谱中心化

    W, H = F.shape
    u, v = np.meshgrid(np.arange(-W//2, W//2), np.arange(-H//2, H//2), indexing='ij') # 生成二维矩阵

    H = np.exp(-k*((u)**2 + (v)**2)**(5/6))
    F = F * H

    X = np.fft.ifftshift(F) # 逆频谱中心化

    # 反变换
    x = np.real(np.fft.ifft2(X))
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    return x, H, F


def turbulence_add_noise(image, k =0.001):
    f = np.float32(image) / 255.0
    F = np.fft.fft2(f) # 转换到频域
    F = np.fft.fftshift(F) # 频谱中心化

    W, H = F.shape
    u, v = np.meshgrid(np.arange(-W // 2, W // 2), np.arange(-H // 2, H // 2), indexing='ij')  # 生成二维矩阵

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("原频谱")
    plt.colorbar()
    plt.show()

    # 退化函数
    H = np.exp(-k*((u)**2 + (v)**2)**(5/6))
    F = F * H

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("退化后的频谱")
    plt.colorbar()
    plt.show()

    # 添加随机噪声
    noise = np.random.rayleigh(scale=50, size=F.shape)
    F = F + noise

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("加了噪声后的频谱")
    plt.colorbar()
    plt.show()

    magnitude_spectrum_noise = np.log(np.abs(noise) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum_noise, cmap='gray')
    plt.title("噪声频谱")
    plt.colorbar()
    plt.show()

    X = np.fft.ifftshift(F) # 逆频谱中心化
    # 反变换
    x = np.real(np.fft.ifft2(X))
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    return x, F


T = 0.5
a = 0.05
b = 0.05

def motion_blur(image):
    f = np.float32(image) / 255.0
    F = np.fft.fft2(f) # 转换到频域
    F = np.fft.fftshift(F) # 频谱中心化

    W, H = F.shape
    u, v = np.meshgrid(np.arange(-W//2, W//2), np.arange(-H//2, H//2), indexing='ij') # 生成二维矩阵

    # 运动模糊函数
    pi_au_bv = np.pi * (u * a + v * b)
    H = np.ones_like(F)
    mask = pi_au_bv != 0
    H[mask] = (T / pi_au_bv[mask]) * np.sin(pi_au_bv[mask]) * np.exp(-1j * pi_au_bv[mask])

    F = F * H

    X = np.fft.ifftshift(F) # 逆频谱中心化

    # 反变换
    x = np.real(np.fft.ifft2(X))
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    return x, H, F


def motion_blur_add_noise(image):
    f = np.float32(image) / 255.0
    F = np.fft.fft2(f)  # 转换到频域
    F = np.fft.fftshift(F)  # 频谱中心化

    W, H = F.shape
    u, v = np.meshgrid(np.arange(-W // 2, W // 2), np.arange(-H // 2, H // 2), indexing='ij')  # 生成二维矩阵

    # 运动模糊函数
    pi_au_bv = np.pi * (u * a + v * b)
    H = np.ones_like(F)
    mask = pi_au_bv != 0 # 防止除0错误
    H[mask] = (T / pi_au_bv[mask]) * np.sin(pi_au_bv[mask]) * np.exp(-1j * pi_au_bv[mask])

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("运动原频谱")
    plt.colorbar()
    plt.show()

    F = F * H

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("退化后的频谱")
    plt.colorbar()
    plt.show()

    # 添加随机噪声
    noise = np.random.rayleigh(scale=50, size=F.shape)
    F = F + noise

    magnitude_spectrum = np.log(np.abs(F) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("加了噪声的频谱")
    plt.colorbar()
    plt.show()

    magnitude_spectrum_noise = np.log(np.abs(noise) + 1)  # 取对数以增强对比度
    plt.imshow(magnitude_spectrum_noise, cmap='gray')
    plt.title("噪声频谱")
    plt.colorbar()
    plt.show()
    X = np.fft.ifftshift(F)  # 逆频谱中心化

    # 反变换
    x = np.real(np.fft.ifft2(X))
    x = np.clip(x, 0, 1)
    x = (x * 255).astype(np.uint8)

    return x, F


countryside_image = cv2.imread("./resources_labwork4/countryside.png", cv2.IMREAD_GRAYSCALE)

countryside_image_turbulence, _, _ = turbulence(countryside_image)
countryside_image_turbulence_addNoise, _ = turbulence_add_noise(countryside_image)
cv2.imwrite("./resources_labwork4/countryside_degenerated_add_noise.png", countryside_image_turbulence_addNoise)
cv2.imwrite("./resources_labwork4/countryside_degenerated.png", countryside_image_turbulence)

digital_image = cv2.imread("./resources_labwork4/digital.png", cv2.IMREAD_GRAYSCALE)

digital_image_motionBlur, _, _ = motion_blur(digital_image)
digital_image_motionBlur_addNoise, _ = motion_blur_add_noise(digital_image)
cv2.imwrite("./resources_labwork4/digital_motionBlur_add_noise.png", digital_image_motionBlur_addNoise)
cv2.imwrite("./resources_labwork4/digital_motionBlur.png", digital_image_motionBlur)
