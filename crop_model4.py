import cv2
import os


def crop_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Could not read the image: {}".format(input_path))

    h, w = img.shape[:2]

    # Proportional points (clockwise)
    left = int(w * 0.15)
    right = int(w * 0.39)
    top = int(h * 0.27)
    bottom = int(h * 0.327)

    x_min = max(0, min(left, right))
    y_min = max(0, min(top, bottom))
    x_max = min(w, max(left, right))
    y_max = min(h, max(top, bottom))

    if x_max <= x_min or y_max <= y_min:
        raise Exception("Invalid crop coordinates for: {}".format(input_path))

    crop = img[y_min:y_max, x_min:x_max]

    if not cv2.imwrite(output_path, crop):
        raise Exception("Failed to write crop: {}".format(output_path))

    return output_path


def _batch_crop_temp():
    base_dir = os.path.dirname(__file__)
    temp_dir = os.path.join(base_dir, "temp")
    out_dir = os.path.join(base_dir, "cropped")

    if not os.path.isdir(temp_dir):
        raise Exception("Temp folder not found: {}".format(temp_dir))

    os.makedirs(out_dir, exist_ok=True)

    jpg_files = sorted(
        f for f in os.listdir(temp_dir) if f.lower().endswith(".jpg")
    )

    if not jpg_files:
        raise Exception("No .jpg files found in: {}".format(temp_dir))

    for idx, filename in enumerate(jpg_files, start=1):
        img_path = os.path.join(temp_dir, filename)
        out_name = "digital_cnh_cropped_{}.jpg".format(idx)
        out_path = os.path.join(out_dir, out_name)
        crop_image(img_path, out_path)
        print(out_path)


if __name__ == "__main__":
    _batch_crop_temp()
