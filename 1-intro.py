import cv2
import matplotlib.pyplot as plt


# Filter
def color_filter(image):
    # Blue (0), Green (1), Red (2)

    # Blue filter
    # image[:, :, 1] = 0  # Remove green
    # image[:, :, 2] = 0  # Remove red

    # Red filter
    image[:, :, 0] = 0  # Remove blue
    image[:, :, 1] = 0  # Remove green
    return image

# Image Filter
def show_image(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image.")
        return

    image = color_filter(image)
    image = cv2.resize(image, (640, 480))

    cv2.imshow("output", image)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")
    plt.show()

# Video Filter
def show_video() -> None:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = color_filter(frame)
        frame = cv2.resize(frame, (1280, 960))

        cv2.imshow("Filtered Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "assets/Peace.png"
    show_image(image_path)
    show_video()
