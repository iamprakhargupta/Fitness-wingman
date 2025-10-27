# run_pose_on_folder.py
"""
Iterate through a folder, run MoveNet pose estimation on each image,
and display pose predictions inline in a Jupyter notebook.
"""
from typing import List, Optional
import os
import glob
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.display import clear_output, display

# Keypoint constants reused from the notebook
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y',
    (5, 11): 'm', (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores: np.ndarray,
                                     height: int, width: int,
                                     keypoint_threshold: float = 0.11):
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(image: np.ndarray,
                             keypoints_with_scores: np.ndarray,
                             crop_region=None,
                             output_image_height: Optional[int] = None) -> np.ndarray:
    """
    Draw keypoints and skeleton on a numpy image (H,W,3) and return the overlaid image.
    Uses Agg backend for robust rendering.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    keypoint_locs, keypoint_edges, edge_colors = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle((xmin, ymin), rec_width, rec_height,
                                 linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    image_from_plot = arr[:, :, :3].copy()
    plt.close(fig)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        import cv2  # local import to avoid heavy dependency if unused elsewhere
        image_from_plot = cv2.resize(image_from_plot, dsize=(output_image_width, output_image_height),
                                     interpolation=cv2.INTER_CUBIC)
    return image_from_plot

# Model loader for MoveNet (SavedModel via TF-Hub)
def load_movenet(model_name: str = "movenet_lightning"):
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = module.signatures['serving_default']

    def movenet_fn(input_image: tf.Tensor) -> np.ndarray:
        # SavedModel expects int32
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        return outputs['output_0'].numpy()

    return movenet_fn, input_size

def list_image_files(folder: str, exts: Optional[List[str]] = None) -> List[str]:
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    files = sorted(set(files))
    return files

def process_folder(folder: str,
                   model_name: str = "movenet_lightning",
                   max_images: Optional[int] = None,
                   pause_seconds: float = 1.0):
    """
    Iterate through images in `folder`, run pose estimation, and display overlays
    inline in a Jupyter notebook. Clears output between images for sequential display.
    """
    movenet, input_size = load_movenet(model_name)
    image_files = list_image_files(folder)
    if max_images:
        image_files = image_files[:max_images]

    if not image_files:
        print(f"No images found in {folder}")
        return

    for img_path in image_files:
        try:
            raw = tf.io.read_file(img_path)
            img = tf.io.decode_image(raw, channels=3, expand_animations=False)
            img = tf.cast(img, tf.uint8).numpy()
        except Exception:
            # skip unreadable files
            continue

        # Prepare model input
        input_image = tf.expand_dims(img, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        # Run model
        keypoints_with_scores = movenet(input_image)

        # Prepare display image and overlay
        display_image = tf.expand_dims(img, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

        # Display inline in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(8, 8 * (overlay.shape[0] / overlay.shape[1])))
        plt.imshow(overlay)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
        display(plt.gcf())
        plt.close()

        time.sleep(pause_seconds)

def process_folder_grid(folder: str,
                        model_name: str = "movenet_lightning",
                        max_images: Optional[int] = None,
                        cols: int = 3,
                        batch_size: Optional[int] = None,
                        pause_seconds: float = 1.0):
    """
    Run pose on images in `folder` and display results in a grid per batch.
    - cols: number of columns in the grid
    - batch_size: number of images per displayed batch; defaults to cols * 2
    """
    from math import ceil
    movenet, input_size = load_movenet(model_name)
    image_files = list_image_files(folder)
    if max_images:
        image_files = image_files[:max_images]
    if not image_files:
        print(f"No images found in {folder}")
        return

    if batch_size is None:
        batch_size = cols * 2

    # process in batches
    for bstart in range(0, len(image_files), batch_size):
        batch_paths = image_files[bstart:bstart + batch_size]
        overlays = []
        titles = []
        for img_path in batch_paths:
            try:
                raw = tf.io.read_file(img_path)
                img = tf.io.decode_image(raw, channels=3, expand_animations=False)
                img = tf.cast(img, tf.uint8).numpy()
            except Exception:
                continue

            # model input
            input_image = tf.expand_dims(img, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
            keypoints_with_scores = movenet(input_image)

            # prepare overlay (smaller resize for grid speed)
            display_image = tf.expand_dims(img, axis=0)
            display_image = tf.cast(tf.image.resize_with_pad(display_image, 512, 512), dtype=tf.int32)
            overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
            overlays.append(overlay)
            titles.append(os.path.basename(img_path))

        if not overlays:
            continue

        rows = ceil(len(overlays) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax in axes_list:
            ax.axis("off")

        for i, (ov, title) in enumerate(zip(overlays, titles)):
            ax = axes_list[i]
            ax.imshow(ov)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # hide unused axes
        for j in range(len(overlays), len(axes_list)):
            axes_list[j].set_visible(False)

        clear_output(wait=True)
        display(fig)
        plt.close(fig)

        time.sleep(pause_seconds)

if __name__ == "__main__":
    # Example usage when running this script directly in a notebook with %run:
    # %run -i run_pose_on_folder.py
    # Then call:
    # process_folder(r"C:\path\to\images", max_images=10, pause_seconds=1.5)
    pass