"""
Turn a canvas drawing or a photo into exactly what the models saw in training.

Training pipeline (digits_classifier.ipynb):
    transforms.ToTensor()               -> uint8 [0, 255] becomes float [0, 1]
    transforms.Normalize((0.5,), (0.5,)) -> (x - 0.5) / 0.5, so [-1, 1]

MNIST images themselves also follow a fixed layout that a raw drawing does not:
    - white digit on a black background
    - digit scaled so its longer side fits a 20x20 box (aspect ratio kept)
    - placed in a 28x28 frame so its center of mass sits at the center
This module reproduces that layout before applying the same normalization.
"""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

MNIST_SIZE = 28
DIGIT_BOX = 20
NORM_MEAN, NORM_STD = 0.5, 0.5  # transforms.Normalize((0.5,), (0.5,))


def canvas_to_gray(rgba: np.ndarray) -> np.ndarray:
    """Canvas RGBA (white ink on black) -> uint8 grayscale, ink = bright."""
    rgba = rgba.astype(np.float32)
    rgb_max = rgba[..., :3].max(axis=-1)
    alpha = rgba[..., 3] / 255.0 if rgba.shape[-1] == 4 else 1.0
    return np.clip(rgb_max * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Uploaded photos
# ---------------------------------------------------------------------------

# Typical MNIST stroke thickness inside the 28x28 frame, measured on the test set.
MNIST_STROKE_PX = 2.8
PHOTO_WORK_SIZE = 800  # enough detail for thin pen strokes, still fast


def _otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = values.size
    cum_count = np.cumsum(hist)
    cum_mean = np.cumsum(hist * np.arange(256))
    w0 = cum_count / total
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = cum_mean / cum_count
        mu1 = (cum_mean[-1] - cum_mean) / (total - cum_count)
        between = w0 * w1 * (mu0 - mu1) ** 2
    return int(np.argmax(np.nan_to_num(between)))


def _stroke_width(mask: np.ndarray) -> float:
    """Median stroke thickness in pixels."""
    dt = ndimage.distance_transform_edt(mask)
    ridge = (dt == ndimage.maximum_filter(dt, size=3)) & mask
    return float(2 * np.median(dt[ridge])) if ridge.any() else 1.0


def _select_digit(mask: np.ndarray) -> np.ndarray:
    """Keep the ink that belongs to the digit; drop shadows, paper edges and specks."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, labels, idx)
    boxes = ndimage.find_objects(labels)
    H, W = mask.shape

    def touches_border(b):
        return b[0].start == 0 or b[1].start == 0 or b[0].stop == H or b[1].stop == W

    def flat_sliver(b):
        # Much wider than tall: a leftover piece of a ruled line, never a whole digit.
        return (b[1].stop - b[1].start) > 4 * (b[0].stop - b[0].start)

    # Blobs touching the photo edge are almost always table, shadow or paper edge.
    # Only drop them if some ink remains elsewhere (a tight crop may touch the edge).
    inner = [i for i in range(n) if not touches_border(boxes[i])]
    candidates = inner if inner else list(range(n))
    shaped = [i for i in candidates if not flat_sliver(boxes[i])]
    main = max(shaped or candidates, key=lambda i: sizes[i])
    if sizes[main] == 0:
        return np.zeros_like(mask)

    mb = boxes[main]
    mh, mw = mb[0].stop - mb[0].start, mb[1].stop - mb[1].start
    reach = 0.6 * max(mh, mw)  # detached parts of the same digit (e.g. the top bar of a 5)

    keep = np.zeros(n + 1, dtype=bool)
    keep[main + 1] = True
    for i in candidates:
        if i == main or sizes[i] < 0.03 * sizes[main]:
            continue
        b = boxes[i]
        dy = max(mb[0].start - b[0].stop, b[0].start - mb[0].stop, 0)
        dx = max(mb[1].start - b[1].stop, b[1].start - mb[1].stop, 0)
        if max(dy, dx) <= reach:
            keep[i + 1] = True
    return keep[labels]


def _long_rows(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Pixels belonging to horizontal runs of True at least min_len long."""
    out = np.zeros_like(mask)
    padded = np.pad(mask, ((0, 0), (1, 1))).astype(np.int8)
    for r in np.flatnonzero(mask.sum(axis=1) >= min_len):
        edges = np.flatnonzero(np.diff(padded[r]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_len:
                out[r, a:b] = True
    return out


def _ruled_lines(mask: np.ndarray) -> np.ndarray:
    """Straight lines crossing most of the photo (notebook rules, grids, margins),
    checked at small angles to allow for a slightly tilted photo."""
    soft = ndimage.binary_closing(mask, structure=np.ones((1, 7)))
    soft = ndimage.maximum_filter(soft, size=(5, 1))
    lines = np.zeros_like(mask)
    for transpose in (False, True):  # horizontal, then vertical lines
        m = soft.T if transpose else soft
        length = int(0.75 * m.shape[1])
        for angle in np.arange(-4.5, 4.6, 0.75):
            rot = ndimage.rotate(m, angle, order=0, reshape=False) if angle else m
            found = _long_rows(rot, length)
            if not found.any():
                continue
            if angle:
                found = ndimage.rotate(found, -angle, order=0, reshape=False)
            lines |= found.T if transpose else found
    return ndimage.binary_dilation(lines, iterations=1) & mask


def photo_to_gray(image: Image.Image) -> np.ndarray:
    """A phone photo or scan of one handwritten digit -> uint8 image, ink = bright.

    Steps: fix phone rotation, remove uneven lighting and shadows, separate ink
    from paper, remove ruled lines, keep only the digit, then redraw its strokes
    at MNIST thickness.
    """
    image = ImageOps.exif_transpose(image)  # phones store rotation as metadata
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(paper, rgba)  # transparent PNGs: ink on white
    gray = image.convert("L")
    gray.thumbnail((PHOTO_WORK_SIZE, PHOTO_WORK_SIZE), Image.LANCZOS)
    g = np.asarray(gray, dtype=np.float32)

    # Most of a photo is paper: a dark median means light ink on a dark surface.
    if np.median(g) < 127:
        g = 255.0 - g

    # Estimate paper brightness everywhere (removes gradients and soft shadows),
    # then measure ink as darkness below it.
    k = max(15, int(0.08 * max(g.shape)))
    background = ndimage.grey_closing(g, size=(k, k))
    background = ndimage.uniform_filter(background, size=k)
    ink = np.clip(background - g, 0, 255)

    ink_u8 = ink.astype(np.uint8)
    t = max(_otsu_threshold(ink_u8), 20)  # never treat faint noise as ink
    mask = ndimage.binary_opening(ink_u8 > t, structure=np.ones((2, 2)))
    lines = _ruled_lines(mask)
    if lines.any():
        # Reconnect digit strokes that a removed line had crossed.
        kept = mask & ~lines
        bridged = ndimage.binary_closing(kept, structure=np.ones((11, 1)))
        bridged |= ndimage.binary_closing(kept, structure=np.ones((1, 11)))
        mask = kept | (lines & bridged)
    mask = _select_digit(mask)
    if mask.sum() < 30:
        return np.zeros_like(ink_u8)

    # Redraw strokes so they end up ~MNIST_STROKE_PX wide after scaling to 20px.
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    scale = DIGIT_BOX / max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
    target = MNIST_STROKE_PX / scale
    width = _stroke_width(mask)
    if width < 0.8 * target:    # thin pen: thicken
        mask = ndimage.distance_transform_edt(~mask) <= (target - width) / 2
    elif width > 1.6 * target:  # fat marker: thin down
        mask = ndimage.distance_transform_edt(mask) > (width - target) / 2

    # Soft edges, like the anti-aliased strokes in MNIST.
    out = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, sigma=0.35 * target / 2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Shared by drawings and photos
# ---------------------------------------------------------------------------

def to_mnist_layout(gray: np.ndarray, ink_threshold: int = 30):
    """uint8 grayscale (ink bright) -> 28x28 uint8 in MNIST layout, or None if blank."""
    mask = gray > ink_threshold
    if mask.sum() < 20:  # nothing meaningful drawn yet
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = gray[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    # Fit the longer side into the 20x20 box, keeping aspect ratio.
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)

    frame = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float32)
    top, left = (MNIST_SIZE - new_h) // 2, (MNIST_SIZE - new_w) // 2
    frame[top: top + new_h, left: left + new_w] = np.asarray(resized, dtype=np.float32)

    # Shift so the center of mass lands on the frame center, as in MNIST.
    cy, cx = ndimage.center_of_mass(frame)
    shift = (MNIST_SIZE / 2 - 0.5 - cy, MNIST_SIZE / 2 - 0.5 - cx)
    frame = ndimage.shift(frame, shift, order=1, mode="constant", cval=0.0)

    return np.clip(frame, 0, 255).astype(np.uint8)


def normalize(mnist_uint8: np.ndarray) -> np.ndarray:
    """Same math as ToTensor() + Normalize((0.5,), (0.5,)). Returns (1, 1, 28, 28) float32."""
    x = mnist_uint8.astype(np.float32) / 255.0      # ToTensor
    x = (x - NORM_MEAN) / NORM_STD                  # Normalize
    return x[None, None, :, :]