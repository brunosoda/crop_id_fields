import cv2
import os
import sys


# =========================
# Masking (black rectangles)
# =========================
def mask_regions(img):
    """
    Aplica retângulos pretos em regiões que você não quer que o LLM enxergue.
    As coordenadas são proporcionais (0..1), para funcionar em qualquer resolução.
    Ajuste os valores conforme o seu layout.

    NOTE: Masking is currently DEACTIVATED in the pipeline (crop-only).
    The lines below are intentionally kept for future use.
    """
    h, w = img.shape[:2]
    out = img.copy()

    # (x1, y1, x2, y2) em porcentagem do tamanho da imagem
    rects = [
        # EXEMPLOS - AJUSTE:
        # 1) Escritas em verde no canto superior esquerdo
        (0.00, 0.00, 0.35, 0.19),

        # 2) Foto 3x4 (lado esquerdo)
        (0.00, 0.33, 0.17, 1.00),

        # 3) Toda informação textual abaixo do número vermelho do lado direito
        (0.65, 0.12, 1.00, 1.00),
    ]

    for (x1p, y1p, x2p, y2p) in rects:
        x1 = int(x1p * w)
        y1 = int(y1p * h)
        x2 = int(x2p * w)
        y2 = int(y2p * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    return out


def crop_image(input_path, output_path, apply_mask=False):
    """
    Crop-only by default. Masking is deactivated (apply_mask=False).
    If you ever want to re-enable masking, call with apply_mask=True.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise Exception("Could not read the image: {}".format(input_path))

    h, w = img.shape[:2]

    # Proportional crop region
    left = int(w * 0.14)
    right = int(w * 0.71)
    top = int(h * 0.35)
    bottom = int(h * 0.47)

    x_min = max(0, min(left, right))
    y_min = max(0, min(top, bottom))
    x_max = min(w, max(left, right))
    y_max = min(h, max(top, bottom))

    if x_max <= x_min or y_max <= y_min:
        raise Exception("Invalid crop coordinates for: {}".format(input_path))

    crop = img[y_min:y_max, x_min:x_max]

    # Masking is deactivated by default; keep function available for later.
    if apply_mask:
        crop = mask_regions(crop)

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

        # Crop-only (masking deactivated)
        crop_image(img_path, out_path, apply_mask=False)

        print(out_path)


if __name__ == "__main__":
    # CLI:
    # python crop_model7.py <input.jpg> [output.jpg]
    # Optional: add --mask if you ever want to re-enable masking:
    # python crop_model7.py <input.jpg> [output.jpg] --mask

    apply_mask = ("--mask" in sys.argv)
    args = [a for a in sys.argv[1:] if a != "--mask"]

    if len(args) >= 1:
        input_path = args[0]

        if len(args) >= 2:
            output_path = args[1]
        else:
            base_dir = os.path.dirname(__file__)
            out_dir = os.path.join(base_dir, "cropped")
            os.makedirs(out_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(out_dir, f"{name}_cropped.jpg")

        crop_image(input_path, output_path, apply_mask=apply_mask)
        print(output_path)
    else:
        _batch_crop_temp()
