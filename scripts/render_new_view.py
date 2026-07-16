import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isar_runner import ISARRunner


# ========================== User settings ==========================
CASE_NAME = "1999JV6"
CONF_PATH = PROJECT_ROOT / "confs" / "isar_fuyan.conf"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "exp" / CASE_NAME / "checkpoints" / "ckpt_002000.pth"
)

# The base frame only supplies range/azimuth axes and the image size.
# Its original viewing direction is replaced by NEW_RADAR_LOS.
BASE_VIEW_INDEX = 0
NEW_RADAR_LOS = [1.0, 0.0, 0.0]  # Direction from target to radar, in object xyz.
ROTATION_AXIS = [0.0, 0.0, 1.0]

# Keep these as None to use the base frame size and validate.n_height.
OUTPUT_HEIGHT = None
OUTPUT_WIDTH = None
N_HEIGHT = None

OUTPUT_DIR = PROJECT_ROOT / "exp" / CASE_NAME / "new_views"
OUTPUT_NAME = "view_los_1_0_0"
PNG_PERCENTILE = 100.0  # 100.0 means per-image maximum normalization.
SAVE_NPY = True
# ==================================================================


def normalize_vector(vector, name):
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.shape != (3,):
        raise ValueError("{} must contain exactly three values".format(name))
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("{} must be a non-zero vector".format(name))
    return value / norm


def clone_frame_meta(frame_meta):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in frame_meta.items()
    }


def image_to_uint8(image, percentile):
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, None)

    percentile = float(percentile)
    if not 0.0 < percentile <= 100.0:
        raise ValueError("PNG_PERCENTILE must be in (0, 100]")
    scale = float(np.percentile(image, percentile))
    if scale <= 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    return (np.clip(image / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def main():
    if not CONF_PATH.is_file():
        raise FileNotFoundError("Config not found: {}".format(CONF_PATH))
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(CHECKPOINT_PATH))

    # Relative paths inside the config are defined from the project root.
    os.chdir(str(PROJECT_ROOT))
    runner = ISARRunner(
        str(CONF_PATH),
        mode="validate_image",
        case=CASE_NAME,
        is_continue=False,
        checkpoint_name=str(CHECKPOINT_PATH),
    )
    if runner.model_type != "nerf":
        raise ValueError(
            "render_new_view.py requires model.type = \"nerf\", got {!r}".format(
                runner.model_type
            )
        )
    if not 0 <= BASE_VIEW_INDEX < runner.dataset.n_images:
        raise IndexError(
            "BASE_VIEW_INDEX {} is outside [0, {})".format(
                BASE_VIEW_INDEX, runner.dataset.n_images
            )
        )

    target_image, frame_meta = runner.dataset.get_frame(BASE_VIEW_INDEX)
    frame_meta = clone_frame_meta(frame_meta)

    los = normalize_vector(NEW_RADAR_LOS, "NEW_RADAR_LOS")
    rot_axis = normalize_vector(ROTATION_AXIS, "ROTATION_AXIS")
    old_los = frame_meta["radar_los"]
    frame_meta["radar_los"] = torch.as_tensor(
        los, dtype=old_los.dtype, device=old_los.device
    )
    runner.renderer.rot_axis = torch.as_tensor(rot_axis, dtype=torch.float32)

    if N_HEIGHT is not None:
        runner.set_renderer_n_height(N_HEIGHT)

    base_height, base_width = target_image.shape
    image_shape = (
        int(OUTPUT_HEIGHT) if OUTPUT_HEIGHT is not None else int(base_height),
        int(OUTPUT_WIDTH) if OUTPUT_WIDTH is not None else int(base_width),
    )

    runner.nerf_network.eval()
    with torch.inference_mode():
        render_out = runner.render_isar(
            frame_meta,
            image_shape=image_shape,
            cos_anneal_ratio=1.0,
        )
    image = render_out["isar"].detach().cpu().numpy().astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "{}.png".format(OUTPUT_NAME)
    npy_path = OUTPUT_DIR / "{}.npy".format(OUTPUT_NAME)

    png_image = image_to_uint8(image, PNG_PERCENTILE)
    if not cv.imwrite(str(png_path), png_image):
        raise IOError("Failed to write image: {}".format(png_path))
    if SAVE_NPY:
        np.save(str(npy_path), image)

    print("Checkpoint : {}".format(CHECKPOINT_PATH))
    print("Iteration  : {}".format(runner.iter_step))
    print("Base view  : {}".format(BASE_VIEW_INDEX))
    print("New LOS    : {}".format(los.tolist()))
    print("Image shape: {}".format(image.shape))
    print("Value range: [{:.6g}, {:.6g}]".format(float(image.min()), float(image.max())))
    print("PNG saved  : {}".format(png_path))
    if SAVE_NPY:
        print("NPY saved  : {}".format(npy_path))


if __name__ == "__main__":
    main()
