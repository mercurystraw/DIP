import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy.ndimage import generic_filter


def tofloat(image):
    return image.astype(np.float64) / 255.0  # 将图像转换为浮点数并归一化


# 局部均值
def localmean(image, nhood=None):
    if nhood is None:
        nhood = np.ones((3, 3)) /9.0
    else:
        nhood = nhood / np.sum(nhood)
    return generic_filter(image, np.mean, footprint=nhood)

# 局部阈值
def localthresh(image, nhood, a, b, meantype='local'):
    image = tofloat(image)
    SIG = generic_filter(image, np.std, footprint=nhood)
    if meantype == 'global':
        MEAN = np.mean(image)
    else:
        MEAN = localmean(image, nhood)
    return (image > a * SIG) & (image > b * MEAN)


I1 = io.imread('./resources_labwork8/4.png')
I1 = cv2.GaussianBlur(I1, (3, 3), 0)  # 高斯模糊
I1 = tofloat(I1)  # 将图像转换为浮点数并归一化

# 计算局部标准差图像
SIG = generic_filter(I1, np.std, footprint=np.ones((3, 3)))
# 利用局部标准差的方法对图像进行分割
g = localthresh(I1, np.ones((3, 3)), 30, 1.5, 'global')


# 显示结果
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(I1, cmap='gray')
plt.title('original image')
plt.axis('off')

plt.subplot(132)
plt.imshow(SIG, cmap='gray')
plt.title('local standard deviation')
plt.axis('off')

plt.subplot(133)
plt.imshow(g, cmap='gray')
plt.title('local thresholding')
plt.axis('off')

plt.show()