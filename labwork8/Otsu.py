import cv2

image = cv2.imread("./resources_labwork8/3.png", cv2.IMREAD_GRAYSCALE)
blur_image = cv2.GaussianBlur(image, (5, 5), 0)

# 应用 Otsu 的全局阈值处理
# src：输入图像（通常是灰度图像）。
# thresh：阈值，用于将图像分割为前景和背景。在 Otsu 方法中，通常设为 0，表示自动计算最佳阈值。
# maxval：当像素值超过阈值时应赋予的值（通常是255，表示前景）。
# cv2.THRESH_BINARY：将大于阈值的像素设为最大值，小于等于阈值的像素设为0。
# cv2.THRESH_OTSU：使用 Otsu 方法自动选择阈值，通常与其他阈值化类型组合使用（如 cv2.THRESH_BINARY）。
thresh_value, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("Threshold value:", thresh_value)
# 显示原始图像和二值化结果
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image (Otsu)', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

