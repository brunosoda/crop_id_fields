import argparse
import os

import cv2
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main():
    parser = argparse.ArgumentParser(
        description="Compare rg2_crop.jpg with another .jpg using SSIM."
    )
    parser.add_argument(
        "other_image",
        help="Path to the other cropped .jpg to compare against rg2_crop.jpg",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize the other image to match rg2_crop.jpg if sizes differ.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use color SSIM by averaging per-channel scores (default: grayscale).",
    )
    args = parser.parse_args()

    base_path = os.path.join(os.path.dirname(__file__), "images", "rg2_crop.jpg")
    base_image = load_image(base_path)
    other_image = load_image(args.other_image)

    if base_image.shape[:2] != other_image.shape[:2]:
        if not args.resize:
            raise ValueError(
                "Image sizes differ. Use --resize to match dimensions."
            )
        other_image = cv2.resize(
            other_image,
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    if args.color:
        channels = cv2.split(base_image)
        other_channels = cv2.split(other_image)
        scores = []
        for base_channel, other_channel in zip(channels, other_channels):
            scores.append(
                ssim(base_channel, other_channel, data_range=255)
            )
        score = sum(scores) / len(scores)
    else:
        base_gray = to_gray(base_image)
        other_gray = to_gray(other_image)
        score = ssim(base_gray, other_gray, data_range=255)

    print("SSIM:", score)


if __name__ == "__main__":
    main()
