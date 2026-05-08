import os
import cv2
from tensorflow.keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()

base_path = "dataset"

for i in range(10):
    os.makedirs(f"{base_path}/{i}", exist_ok=True)

count_per_digit = 1000
saved_counts = {i: 0 for i in range(10)}

for image, label in zip(x_train, y_train):

    if saved_counts[label] < count_per_digit:

        filename = f"{base_path}/{label}/{saved_counts[label]}.png"

        cv2.imwrite(filename, image)

        saved_counts[label] += 1

    if all(count == count_per_digit for count in saved_counts.values()):
        break

print("Dataset saved successfully.")