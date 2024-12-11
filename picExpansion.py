import cv2
import os
import numpy as np

def augment_image(image):
    augmented_images = []

    # 回転
    rows, cols = image.shape[:2]
    for angle in [-10, 10]:
        matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, matrix, (cols, rows))
        augmented_images.append(rotated)

    # フリップ
    flipped = cv2.flip(image, 1)  # 左右反転
    augmented_images.append(flipped)

    # 明るさ調整
    brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(brightened)

    return augmented_images

# 使用例
input_folder = "white_lines_output"
output_folder = "augmented_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        augmented = augment_image(image)

        for i, aug_img in enumerate(augmented):
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug{i}.jpg")
            cv2.imwrite(output_path, aug_img)

