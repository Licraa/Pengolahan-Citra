# Author: [Raja AryansahPutra , Analisari Putri]
# Date: 2023-10-01
# Description: Program ini melakukan Histogram Equalization dan Histogram Specification pada citra input.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Fungsi untuk melakukan Histogram Equalization
def histogram_equalization(image):
    # Konversi ke grayscale jika citra berwarna
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Lakukan histogram equalization
    equalized = cv2.equalizeHist(gray)
    return gray, equalized

# Fungsi untuk melakukan Histogram Specification
def histogram_specification(source, reference):
    # Hitung histogram kumulatif dari source dan reference
    def calculate_cdf(hist):
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()  # Normalisasi
        return cdf_normalized

    # Konversi ke grayscale jika citra berwarna
    if len(source.shape) == 3:
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    else:
        source_gray = source

    if len(reference.shape) == 3:
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference

    # Hitung histogram
    source_hist, _ = np.histogram(source_gray.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_gray.flatten(), 256, [0, 256])

    # Hitung CDF
    source_cdf = calculate_cdf(source_hist)
    reference_cdf = calculate_cdf(reference_hist)

    # Membuat mapping antara source dan reference
    mapping = np.zeros(256)
    for i in range(256):
        diff = np.abs(source_cdf[i] - reference_cdf)
        mapping[i] = np.argmin(diff)

    # Transformasikan citra source menggunakan mapping
    height, width = source_gray.shape
    matched_image = np.zeros_like(source_gray)
    for y in range(height):
        for x in range(width):
            matched_image[y, x] = mapping[source_gray[y, x]]

    return matched_image

# Main program
if __name__ == "__main__":
    # Sembunyikan jendela utama tkinter
    Tk().withdraw()

    # Pilih file citra input
    print("Pilih citra input...")
    input_image_path = askopenfilename(
        title="Pilih Citra Input",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    # Pilih file citra referensi
    print("Pilih citra referensi...")
    reference_image_path = askopenfilename(
        title="Pilih Citra Referensi",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not input_image_path or not reference_image_path:
        print("Error: Anda harus memilih kedua file!")
        exit()

    # Baca citra
    input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)

    if input_image is None or reference_image is None:
        print("Error: Citra tidak ditemukan atau tidak valid!")
        exit()

    # Histogram Equalization
    original, equalized = histogram_equalization(input_image)

    # Histogram Specification
    matched_image = histogram_specification(input_image, reference_image)

    # Plot semua hasil dalam satu halaman
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Baris 1: Original Image
    axes[0, 0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].hist(original.ravel(), 256, [0, 256], color="gray")
    axes[0, 1].set_title("Histogram Original")
    axes[0, 1].set_xlabel("Pixel Intensity")
    axes[0, 1].set_ylabel("Frequency")

    axes[0, 2].imshow(original, cmap="gray")
    axes[0, 2].set_title("Grayscale Original")
    axes[0, 2].axis("off")

    # Baris 2: Equalized Image
    axes[1, 0].imshow(equalized, cmap="gray")
    axes[1, 0].set_title("Equalized Image")
    axes[1, 0].axis("off")

    axes[1, 1].hist(equalized.ravel(), 256, [0, 256], color="gray")
    axes[1, 1].set_title("Histogram Equalized")
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Frequency")

    axes[1, 2].imshow(equalized, cmap="gray")
    axes[1, 2].set_title("Grayscale Equalized")
    axes[1, 2].axis("off")

    # Baris 3: Matched Image
    axes[2, 0].imshow(matched_image, cmap="gray")
    axes[2, 0].set_title("Specification Image")
    axes[2, 0].axis("off")

    axes[2, 1].hist(matched_image.ravel(), 256, [0, 256], color="gray")
    axes[2, 1].set_title("Histogram Specification")
    axes[2, 1].set_xlabel("Pixel Intensity")
    axes[2, 1].set_ylabel("Frequency")

    axes[2, 2].imshow(matched_image, cmap="gray")
    axes[2, 2].set_title("Grayscale Matched")
    axes[2, 2].axis("off")

    # Atur layout agar rapi
    plt.tight_layout()
    plt.show()