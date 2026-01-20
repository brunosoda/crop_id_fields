import argparse
import glob
import os

import cv2
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_ssim(img_a, img_b, use_color=False):
    if use_color:
        a_ch = cv2.split(img_a)
        b_ch = cv2.split(img_b)
        scores = []
        for ca, cb in zip(a_ch, b_ch):
            scores.append(ssim(ca, cb, data_range=255))
        return sum(scores) / len(scores)
    else:
        ga = to_gray(img_a)
        gb = to_gray(img_b)
        return ssim(ga, gb, data_range=255)


def main():
    parser = argparse.ArgumentParser(
        description="Compare a cropped image against mask_1.jpg..mask_7.jpg using SSIM."
    )
    parser.add_argument(
        "cropped_image",
        help="Path to the cropped .jpg to compare against all masks.",
    )
    parser.add_argument(
        "--masks-dir",
        default=os.path.join(os.path.dirname(__file__), "masks"),
        help="Directory containing mask_*.jpg files (default: ./masks).",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize the cropped image to match each mask if sizes differ.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use color SSIM (default: grayscale).",
    )
    args = parser.parse_args()

    cropped = load_image(args.cropped_image)

    pattern = os.path.join(args.masks_dir, "mask_*.jpg")
    mask_paths = sorted(glob.glob(pattern))

    # Also allow .JPG
    if not mask_paths:
        pattern2 = os.path.join(args.masks_dir, "mask_*.JPG")
        mask_paths = sorted(glob.glob(pattern2))

    if not mask_paths:
        raise FileNotFoundError(f"No mask_*.jpg found in: {args.masks_dir}")

    results = []
    best = {"mask": None, "model": None, "score": -1.0}

    for mp in mask_paths:
        mask = load_image(mp)

        comp = cropped
        if mask.shape[:2] != cropped.shape[:2]:
            if not args.resize:
                raise ValueError(
                    f"Size differs for {os.path.basename(mp)}: "
                    f"mask={mask.shape[:2]} cropped={cropped.shape[:2]}. Use --resize."
                )
            comp = cv2.resize(
                cropped,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        score = compute_ssim(mask, comp, use_color=args.color)

        # Extract model number from filename mask_6.jpg -> "6"
        base = os.path.basename(mp)
        model = base.replace("mask_", "").split(".")[0]

        results.append((model, base, score))
        if score > best["score"]:
            best = {"mask": base, "model": model, "score": score}

    # Print per-mask scores
    for model, base, score in results:
        print(f"mask {model} ({base}) SSIM: {score:.4f}")

    print(f"\nBEST: model={best['model']} mask={best['mask']} SSIM={best['score']:.4f}")


if __name__ == "__main__":
    main()
