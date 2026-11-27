"""Experiment 1 stimuli: noisy images of blue and yellow circles (port of generate_circles.py).

The paper's 20 image pairs are stored in data/stimuli/ (the original run was unseeded,
so re-running this produces *different* images). Player 1 images are mildly degraded;
player 2 images have two more circles of each colour (same answer) and heavier noise:
more blur, noise, occlusions, colour/contrast jitter, and a small rotation.
Requires Pillow (``pip install -e .[collect]``).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

# Degradation per image set: "normal" for player 1, "cooked" (the original's name) for player 2.
EFFECTS = {
    "normal": {"blur": (1.0, 2.5), "noise": 20, "occlusions": (2, 5), "alpha": 100, "distort": False},
    "cooked": {"blur": (4.0, 6.0), "noise": 50, "occlusions": (6, 10), "alpha": 180, "distort": True},
}


def add_adversarial_effects(image, circles, radius, intensity="normal"):
    """Blur, pixel noise, semi-transparent occluding rectangles that avoid the circles, and for
    "cooked" also colour, rotation and contrast jitter (random draws in the original's order)."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    fx = EFFECTS[intensity]
    image = image.filter(ImageFilter.GaussianBlur(random.uniform(*fx["blur"])))
    noise = np.random.randint(-fx["noise"], fx["noise"], (image.height, image.width, 3), dtype="int")
    image = Image.fromarray(np.clip(np.array(image, dtype="int") + noise, 0, 255).astype("uint8"))
    if fx["distort"]:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))

    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(random.randint(*fx["occlusions"])):
        for _attempt in range(10):
            x1 = random.randint(0, image.width - 50)
            y1 = random.randint(0, image.height - 50)
            x2 = x1 + random.randint(20, 50)
            y2 = y1 + random.randint(20, 50)
            covers_circle = any(
                (cx - radius < x1 < cx + radius and cy - radius < y1 < cy + radius)
                or (cx - radius < x2 < cx + radius and cy - radius < y2 < cy + radius)
                for cx, cy in circles)
            if not covers_circle:
                draw.rectangle([x1, y1, x2, y2], fill=(random.randint(50, 200), random.randint(50, 200),
                                                       random.randint(50, 200), fx["alpha"]))
                break

    if fx["distort"]:
        image = image.rotate(random.uniform(-5, 5))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def circles_image(n_blue: int, n_yellow: int, image_size=(500, 500), radius=20, intensity="normal"):
    from PIL import Image, ImageDraw

    width, height = image_size
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    circles: list[tuple[int, int]] = []

    def free_position():
        for _ in range(1000):
            pos = (random.randint(radius, width - radius), random.randint(radius, height - radius))
            if all(math.dist(c, pos) >= 2 * radius for c in circles):
                return pos
        raise ValueError("Could not place circles without overlapping. Reduce the number of circles.")

    for colour, count in (("blue", n_blue), ("yellow", n_yellow)):
        for _ in range(count):
            pos = free_position()
            circles.append(pos)
            draw.ellipse([pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius],
                         fill=colour, outline="black")
    return add_adversarial_effects(image, circles, radius, intensity=intensity)


def generate(out_dir: Path, n_pairs: int = 20, seed: int | None = None) -> None:
    """Write n_pairs unique (blue, yellow) count pairs to out_dir/player_1 and out_dir/player_2."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    p1, p2 = out_dir / "player_1", out_dir / "player_2"
    for d in (p1, p2):
        d.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, int]] = set()
    while len(seen) < n_pairs:
        x, y = random.randint(1, 10), random.randint(1, 10)   # x blue, y yellow
        if (x, y) in seen:
            continue
        seen.add((x, y))
        circles_image(x, y, intensity="normal").save(p1 / f"{y}_yellow_{x}_blue.png")
        circles_image(x + 2, y + 2, intensity="cooked").save(p2 / f"{y + 2}_yellow_{x + 2}_blue.png")
