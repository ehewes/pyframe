from __future__ import annotations

import math
import os

from PIL import Image


def merge_to_grid(images, images_per_row: int = 2, target_size: int = 2560):
    if not images:
        raise ValueError("No images to merge")

    resized = []
    for img in images:
        img = img.convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        resized.append(img)

    max_width = max(img.size[0] for img in resized)
    max_height = max(img.size[1] for img in resized)
    num_rows = math.ceil(len(resized) / images_per_row)

    grid = Image.new("RGB", (images_per_row * max_width, num_rows * max_height), "black")
    for idx, img in enumerate(resized):
        x = (idx % images_per_row) * max_width
        y = (idx // images_per_row) * max_height
        grid.paste(img, (x, y))
    return grid


def merge_images_to_grid(image_paths, output_path, images_per_row: int = 2, target_size: int = 2560):
    images = [Image.open(path) for path in image_paths]
    try:
        grid = merge_to_grid(images, images_per_row, target_size)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        grid.save(output_path, quality=95)
    finally:
        for img in images:
            img.close()
    return output_path
