import cv2
import numpy as np

# ---------------- Transformasi Intensitas ----------------

# Citra Negatif
def negative(image):
    result = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result [i, j] = 255 - image[i, j]

    return result

# Transformasi Logaritmik
def logaritmic(image):
    result = np.empty_like(image, dtype=np.uint8)

    L = 256
    c = L / np.log(L + 1)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value = image[i, j]
            s = c * np.log(1.0 + value)
            result[i, j] = min(255, max(0, int(s + 0.5)))

    return result

# Transformasi Pangkat (koreksi gamma)
def gamma(image, gammav=0.4):
    result = np.empty_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value = image[i, j] / 255.0
            s = 255.0 * (value ** gammav)
            result[i, j] = int(s + 0.5)

    return result

# Transformasi linear sepotong - sepotong (contrast stretching)
def contrast_stretching(image, r1=80, s1=20, r2=175, s2=240):
    result = np.empty_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = image[i, j]

            if r < r1:
                s = (s1 / r1) * r
            elif r < r2:
                s = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
            else:
                s = ((255 - s2) / (255 - r2)) * (r - r2) + s2

            result[i, j] = min(255, max(0, int(s + 0.5)))

    return result

# Thresholding
def thresholding(image, T=128):
    result = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] < T:
                result[i, j] = 0
            else:
                result[i, j] = 255

    return result


# ---------------- Ekualisasi Histogram ----------------

def histogram(image):
    histogram = np.zeros(256, dtype=np.int32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            n = image[i, j]
            histogram[n] += 1

    return histogram

def histogram_equalization(image):
    result = np.empty_like(image)

    # P(rk) = n(k) / (M x N)
    # cdf = sigmaP
    n = histogram(image)
    mapping = np.zeros(256, dtype=np.uint8)
    MxN = image.shape[0] * image.shape[1]
    cdf = 0

    for k in range(256):
        P = n[k] / MxN
        cdf += P

        mapping[k] = int(255.0 * cdf + 0.5)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            k = image[i, j]
            result[i, j] = mapping[k]

    return result

# ---------------- Main ----------------

def show_result(title, image):
    cv2.imshow (title, image)

def main():
    image_path = "assets/Banner.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not load the image")
        return

    show_result("Original", image)
    show_result("Citra Negatif", negative(image))
    show_result("Transformasi Logaritmik", logaritmic(image))
    show_result("Transformasi Pangkat (koreksi gamma)", gamma(image))
    show_result("Transformasi linear sepotong - sepotong (contrast stretching)", contrast_stretching(image))
    show_result("Thresholding", thresholding(image))
    show_result("Ekualisasi Histogram", histogram_equalization(image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()