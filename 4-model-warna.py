import cv2
import numpy as np


# ---------------- RGB to CMYK ----------------

def rgb_to_cmyk(image):
    rgb = image.astype(np.float64) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    k = 1.0 - np.maximum(np.maximum(r, g), b)

    denominator = 1.0 - k

    c = np.divide(
        1.0 - r - k,
        denominator,
        out=np.zeros_like(r),
        where=denominator != 0
    )

    m = np.divide(
        1.0 - g - k,
        denominator,
        out=np.zeros_like(g),
        where=denominator != 0
    )

    y = np.divide(
        1.0 - b - k,
        denominator,
        out=np.zeros_like(b),
        where=denominator != 0
    )

    return c, m, y, k


# ---------------- RGB to HSI ----------------

def rgb_to_hsi(image):
    rgb = image.astype(np.float64) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    numerator = 0.5 * ((r - g) + (r - b))

    denominator = np.sqrt(
        (r - g) ** 2 +
        (r - b) * (g - b)
    )

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12
    )

    theta = np.degrees(
        np.arccos(np.clip(ratio, -1.0, 1.0))
    )

    h = np.where(
        b <= g,
        theta,
        360.0 - theta
    )

    total = r + g + b
    minimum = np.minimum(np.minimum(r, g), b)

    s = 1.0 - np.divide(
        3.0 * minimum,
        total,
        out=np.zeros_like(total),
        where=total != 0
    )

    i = total / 3.0

    return h, s, i


# ---------------- RGB to HSV ----------------

def rgb_to_hsv(image):
    rgb = image.astype(np.float64) / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    delta = maximum - minimum

    # Value
    v = maximum

    # Saturation
    s = np.divide(
        delta,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum != 0
    )

    # Hue
    h = np.zeros_like(maximum)

    nonzero = delta != 0

    red_max = (maximum == r) & nonzero
    green_max = (maximum == g) & nonzero
    blue_max = (maximum == b) & nonzero

    h[red_max] = (
        60.0 * ((g[red_max] - b[red_max]) / delta[red_max])
    ) % 360.0

    h[green_max] = (
        60.0 * (
            (b[green_max] - r[green_max]) /
            delta[green_max] + 2.0
        )
    )

    h[blue_max] = (
        60.0 * (
            (r[blue_max] - g[blue_max]) /
            delta[blue_max] + 4.0
        )
    )

    return h, s, v


# ---------------- Visualization ----------------

def normalize_channel(channel):
    return np.clip(channel * 255.0, 0, 255).astype(np.uint8)


def show_result(title, image):
    cv2.imshow(title, image)


# ---------------- Main ----------------

def main():
    image_path = "assets/Banner.png"

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Could not load the image")
        return

    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RGB -> CMYK
    c, m, y, k = rgb_to_cmyk(rgb)

    # RGB -> HSI
    h_hsi, s_hsi, i_hsi = rgb_to_hsi(rgb)

    # RGB -> HSV
    h_hsv, s_hsv, v_hsv = rgb_to_hsv(rgb)

    show_result("Original RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    show_result("CMYK - C", normalize_channel(c))
    show_result("CMYK - M", normalize_channel(m))
    show_result("CMYK - Y", normalize_channel(y))
    show_result("CMYK - K", normalize_channel(k))
    show_result("HSI - H", normalize_channel(h_hsi / 360.0))
    show_result("HSI - S", normalize_channel(s_hsi))
    show_result("HSI - I", normalize_channel(i_hsi))
    show_result("HSV - H", normalize_channel(h_hsv / 360.0))
    show_result("HSV - S", normalize_channel(s_hsv))
    show_result("HSV - V", normalize_channel(v_hsv))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()