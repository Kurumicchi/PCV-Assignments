import cv2
import numpy as np


# ---------------- Filter Lowpass ----------------

# Filter Box
def box_filter(image):
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    return cv2.filter2D(
        image,
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE
    )

# Filter Gaussian
def gaussian_filter(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16.0

    return cv2.filter2D(
        image,
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE
    )

# Filter Median
def median_filter(image):
    return cv2.medianBlur(image, 3)


# ---------------- Filter Highpass ----------------

# Penajaman dengan Laplacian
def laplacian_sharpening(image):
    image_float = image.astype(np.float64)

    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    laplacian = cv2.filter2D(
        image_float,
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE
    )

    result = image_float - laplacian
    return np.clip(result, 0, 255).astype(np.uint8)

# Unsharp Masking
def unsharp_masking(image):
    image_float = image.astype(np.float64)

    blurred = cv2.blur(image_float, (3, 3))
    mask = image_float - blurred
    result = image_float + mask

    return np.clip(result, 0, 255).astype(np.uint8)

# Highboost
def highboost(image, k=2.5):
    image_float = image.astype(np.float64)

    blurred = cv2.blur(image_float, (3, 3))
    mask = image_float - blurred
    result = image_float + k * mask

    return np.clip(result, 0, 255).astype(np.uint8)

# Sobel
def sobel_filter(image):
    image_float = image.astype(np.float64)

    gx = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)

    result = np.abs(gx) + np.abs(gy)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------- Main ----------------

def show_result(title, image):
    cv2.imshow(title, image)

def main():
    image_path = "assets/Banner.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not load the image")
        return

    show_result("Original", image)
    show_result("Box Filter", box_filter(image))
    show_result("Gaussian Filter", gaussian_filter(image))
    show_result("Median Filter", median_filter(image))
    show_result("Laplacian Sharpening", laplacian_sharpening(image))
    show_result("Unsharp Masking", unsharp_masking(image))
    show_result("Highboost", highboost(image))
    show_result("Sobel", sobel_filter(image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()