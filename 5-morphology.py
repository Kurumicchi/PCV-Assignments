import cv2
import numpy as np


# ---------------- Morphology ----------------

# Erosi
def erosion(image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.erode(
        image,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# Dilasi
def dilation(image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.dilate(
        image,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# Opening
def opening(image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# Closing
def closing(image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

# ---------------- Main ----------------

def show_result(title, image):
    cv2.imshow(title, image)

def main():
    image_path = "assets/Banner.png"

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print("Error: Could not load the image")
        return

    _, binary = cv2.threshold(
        image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    show_result("Original", image)
    show_result("Binary", binary)
    show_result("Erosion", erosion(binary))
    show_result("Dilation", dilation(binary))
    show_result("Opening", opening(binary))
    show_result("Closing", closing(binary))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()