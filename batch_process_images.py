import glob
import importlib.util
import json
import os
import sys

import boto3
import cv2
import requests
from skimage.metrics import structural_similarity as ssim


# Config
BUCKET_NAME = "conference-core-backend-prod"
PREFIX = "audit_test/"
JSON_FILENAME = "input.json"
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
MASKS_DIR = os.path.join(os.path.dirname(__file__), "masks")
MAX_ROWS = 50
OPTIONAL_HEADERS = {}
USE_PRESIGNED_URLS = True
PRESIGNED_URL_EXPIRY_SECONDS = 3600
DEDUPLICATE_BY_CONFERENCE_UUID = True


def _read_json_rows(json_path):
    rows = []
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
        # Handle both array of objects and single object
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            raise ValueError("JSON must be an array of objects or a single object")

        for item in items:
            conference_uuid = (item.get("conference_uuid") or "").strip()
            file_url = (item.get("file_url") or "").strip()
            if not conference_uuid or not file_url:
                continue
            rows.append(
                {
                    "conference_uuid": conference_uuid,
                    "file_url": file_url,
                }
            )

    if DEDUPLICATE_BY_CONFERENCE_UUID:
        seen = set()
        deduped = []
        for row in rows:
            if row["conference_uuid"] in seen:
                continue
            seen.add(row["conference_uuid"])
            deduped.append(row)
        rows = deduped

    return rows[:MAX_ROWS]


def _prepare_temp_dir(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for filename in os.listdir(temp_dir):
        path = os.path.join(temp_dir, filename)
        if os.path.isfile(path):
            os.remove(path)


def _download_image(url, dest_path):
    response = requests.get(url, headers=OPTIONAL_HEADERS, timeout=30)
    response.raise_for_status()
    with open(dest_path, "wb") as handle:
        handle.write(response.content)


def _s3_cleanup_prefix(s3_client, bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    to_delete = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            to_delete.append({"Key": obj["Key"]})
            if len(to_delete) == 1000:
                s3_client.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
                to_delete = []
    if to_delete:
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})


def _upload_image(s3_client, bucket, key, file_path):
    s3_client.upload_file(
        file_path,
        bucket,
        key,
        ExtraArgs={"ContentType": "image/jpeg"},
    )


def _build_output_url(s3_client, bucket, key):
    if USE_PRESIGNED_URLS:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
        )
    return "https://{}.s3.amazonaws.com/{}".format(bucket, key)


def _write_outputs(base_dir, output_rows):
    json_path = os.path.join(base_dir, "output.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=True, indent=2)


def _find_crop_models(base_dir):
    """Find all crop_model*.py files in the base directory."""
    pattern = os.path.join(base_dir, "crop_model*.py")
    model_files = sorted(glob.glob(pattern))
    models = []
    for model_file in model_files:
        # Extract model number from filename (e.g., crop_model1.py -> 1)
        basename = os.path.basename(model_file)
        model_num = basename.replace("crop_model", "").replace(".py", "")
        models.append((model_num, model_file))
    return models


def _load_crop_module(model_file):
    """Dynamically load a crop model module."""
    module_name = os.path.basename(model_file).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare_ssim(image1_path, image2_path, resize=True):
    """Compare two images using SSIM (grayscale)."""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None:
        raise FileNotFoundError("Could not read image: {}".format(image1_path))
    if img2 is None:
        raise FileNotFoundError("Could not read image: {}".format(image2_path))

    # Resize if needed
    if resize and img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(
            img2,
            (img1.shape[1], img1.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    score = ssim(gray1, gray2, data_range=255)
    return score


def _di_type_from_model(model_num):
    """
    Map best model to di_type_optimized:
    - model_1..5 -> CNH
    - model_6 -> CIN
    - model_7 -> RG
    """
    try:
        n = int(str(model_num).strip())
    except Exception:
        return None

    if n in (1, 2, 3, 4, 5):
        return "CNH"
    if n == 6:
        return "CIN"
    if n == 7:
        return "RG"
    return None


def _find_best_crop(input_path, base_dir, conference_uuid):
    """Try all crop models and return (best_crop_path, best_model_num, best_score)."""
    models = _find_crop_models(base_dir)

    if not models:
        raise Exception("No crop_model*.py files found in {}".format(base_dir))

    best_score = -1.0
    best_crop_path = None
    best_model_num = None
    temp_crops = []

    for model_num, model_file in models:
        try:
            crop_module = _load_crop_module(model_file)

            crop_path = os.path.join(
                TEMP_DIR, "{}_model{}_cropped.jpg".format(conference_uuid, model_num)
            )
            temp_crops.append(crop_path)

            crop_module.crop_image(input_path, crop_path)

            mask_image_path = os.path.join(MASKS_DIR, "mask_{}.jpg".format(model_num))
            if not os.path.isfile(mask_image_path):
                mask_image_path = os.path.join(MASKS_DIR, "mask_{}.JPG".format(model_num))
            if not os.path.isfile(mask_image_path):
                print("Warning: Mask image not found: mask_{}.jpg/JPG".format(model_num))
                continue

            score = _compare_ssim(mask_image_path, crop_path, resize=True)
            print("Model {} similarity: {:.4f}".format(model_num, score))

            if score > best_score:
                best_score = score
                best_crop_path = crop_path
                best_model_num = model_num

        except Exception as exc:
            print("Error processing model {}: {}".format(model_num, exc))
            continue

    if best_crop_path is None:
        raise Exception("No valid crop model produced a result")

    for crop_path in temp_crops:
        if crop_path != best_crop_path and os.path.isfile(crop_path):
            os.remove(crop_path)
            print("Deleted: {}".format(crop_path))

    print("Best model: {} (similarity: {:.4f})".format(best_model_num, best_score))
    return best_crop_path, best_model_num, best_score


def main():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, JSON_FILENAME)
    if not os.path.isfile(json_path):
        print("JSON not found: {}".format(json_path))
        return 1

    rows = _read_json_rows(json_path)
    if not rows:
        print("No valid rows to process in JSON.")
        return 1

    _prepare_temp_dir(TEMP_DIR)

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    s3_client = boto3.client("s3", region_name=region)

    print("Cleaning S3 prefix: s3://{}/{}".format(BUCKET_NAME, PREFIX))
    _s3_cleanup_prefix(s3_client, BUCKET_NAME, PREFIX)

    output_rows = []
    failures = []

    for row in rows:
        conference_uuid = row["conference_uuid"]
        file_url = row["file_url"]
        input_path = os.path.join(TEMP_DIR, "{}.jpg".format(conference_uuid))
        output_path = os.path.join(TEMP_DIR, "{}_cropped.jpg".format(conference_uuid))
        key = "{}{}_cropped.jpg".format(PREFIX, conference_uuid)

        try:
            print("Downloading {}...".format(file_url))
            _download_image(file_url, input_path)

            best_crop_path, best_model_num, best_score = _find_best_crop(
                input_path, base_dir, conference_uuid
            )

            # Rename best crop to the final output path if different
            if best_crop_path != output_path:
                if os.path.isfile(output_path):
                    os.remove(output_path)
                os.rename(best_crop_path, output_path)

            _upload_image(s3_client, BUCKET_NAME, key, output_path)
            cropped_url = _build_output_url(s3_client, BUCKET_NAME, key)

            di_type_optimized = _di_type_from_model(best_model_num)

            output_rows.append(
                {
                    "conference_uuid": conference_uuid,
                    "cropped_file_url": cropped_url,
                    "di_type_optimized": di_type_optimized,
                    "best_model": str(best_model_num),
                    "ssim_score": round(float(best_score), 4),
                }
            )
            print("Uploaded: {}".format(key))
        except Exception as exc:
            failures.append({"conference_uuid": conference_uuid, "error": str(exc)})
            print("Failed {}: {}".format(conference_uuid, exc))

    _write_outputs(base_dir, output_rows)
    print(json.dumps(output_rows, ensure_ascii=True))

    if failures:
        print("Failures: {}".format(len(failures)))
        for failure in failures:
            print("{} -> {}".format(failure["conference_uuid"], failure["error"]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
