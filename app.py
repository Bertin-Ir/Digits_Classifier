"""
Digits Classifier — live Streamlit demo.

Draw a digit (or upload a photo of one) and the CNN trained in
digits_classifier.ipynb predicts it.
"""
import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from model import CNN_model, MLP, load_model
from preprocessing import canvas_to_gray, normalize, photo_to_gray, to_mnist_layout

ROOT = Path(__file__).parent
CNN_WEIGHTS = ROOT / "weights" / "best_mnist_cnn_model.pth"
MLP_WEIGHTS = ROOT / "weights" / "best_mnist_model.pth"

CANVAS_SIZE = 280         # 10x the 28px MNIST frame
STROKE_WIDTH = 20         # gives ~2px strokes after downscaling, like MNIST
LOW_CONFIDENCE = 0.60

MODEL_INFO = {
    "CNN": {"cls": CNN_model, "path": CNN_WEIGHTS, "acc": "99.33%"},
    "MLP": {"cls": MLP, "path": MLP_WEIGHTS, "acc": "97.56%"},
}

st.set_page_config(page_title="Digits Classifier", page_icon="✍️", layout="wide")
torch.set_num_threads(1)


# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800&display=swap');

:root {
  --paper: #F2F5F8;
  --grid: #DCE4ED;
  --ink: #13233F;
  --muted: #5B6B82;
  --pen: #1F5FD1;
  --bar: #C9D4E1;
  --display: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif;
  --body: 'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
}

.stApp {
  background-color: var(--paper);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 28px 28px;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp button {
  font-family: var(--body);
  color: var(--ink);
}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span {
  color: #FFFFFF !important;
  font-weight: 700;
}
.block-container { max-width: 1080px; padding-top: 2.5rem; }
header[data-testid="stHeader"] { background: transparent; }

.hero h1 {
  font-family: var(--display);
  font-weight: 800;
  font-size: clamp(2.2rem, 5vw, 3.4rem);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0 0 .5rem 0;
  color: var(--ink);
}
.hero p { font-size: 1.1rem; color: var(--muted); max-width: 60ch; margin: 0; }

.sheet {
  background: #FFFFFF;
  border: 1px solid var(--grid);
  border-radius: 14px;
  padding: 1.25rem 1.4rem 1.4rem;
}
.sheet h3 {
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.15rem;
  margin: 0 0 .9rem 0;
  padding: 0;
  color: var(--ink);
}
.sheet h3 .acc {
  display: block;
  font-family: var(--body);
  font-weight: 400;
  font-size: .88rem;
  color: var(--muted);
  margin-top: .15rem;
}
.sheet.small { padding: 1rem 1.1rem 1.2rem; }
.sheet.small .readout { flex-direction: column; align-items: flex-start; gap: .5rem; }
.sheet.small .verdict { padding-bottom: 0; }

.readout { display: flex; align-items: flex-end; gap: 1.25rem; margin-bottom: 1rem; }
.digit {
  font-family: var(--display);
  font-weight: 800;
  font-size: 8.5rem;
  line-height: .82;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.digit.empty { color: var(--bar); }
.digit.small { font-size: 5.5rem; }
.verdict { padding-bottom: .4rem; }
.verdict .conf { font-size: 1.6rem; font-weight: 700; color: var(--pen); }
.verdict .note { font-size: .95rem; color: var(--muted); max-width: 26ch; }

.seen { display: flex; gap: 1rem; align-items: center; margin: 1rem 0 0; }
.seen img {
  width: 84px !important; height: 84px !important; flex: none;
  image-rendering: pixelated;
  border-radius: 6px;
  background: #000;
}
.seen p { font-size: .9rem; color: var(--muted); margin: 0; max-width: 34ch; }

.bars { display: grid; gap: .32rem; }
.bar-row { display: grid; grid-template-columns: 1.2rem 1fr 3.4rem; align-items: center; gap: .6rem; }
.bar-row .d { font-family: var(--display); font-weight: 700; text-align: center; }
.bar-track { height: .7rem; background: var(--paper); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--bar); border-radius: 99px; transition: width .25s ease; }
.bar-row.top .bar-fill { background: var(--pen); }
.bar-row .v { font-size: .85rem; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.bar-row.top .v { color: var(--ink); font-weight: 700; }

.hint { color: var(--muted); font-size: .95rem; margin-top: .6rem; }
.foot { color: var(--muted); font-size: .9rem; margin-top: 2.5rem; }
.foot a { color: var(--pen); }

@media (prefers-reduced-motion: reduce) { .bar-fill { transition: none; } }
@media (max-width: 640px) { .digit { font-size: 6rem; } .digit.small { font-size: 4.2rem; } }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model weights…")
def get_model(name: str):
    info = MODEL_INFO[name]
    return load_model(info["cls"], info["path"], device="cpu")


@torch.no_grad()
def predict(model, mnist_uint8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(normalize(mnist_uint8))
    logits = model(x)
    return torch.softmax(logits, dim=1)[0].numpy()


def html(markup: str):
    """Render raw HTML. Lines are flattened so Markdown never mistakes
    indented HTML for a code block."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def png_data_uri(mnist_uint8: np.ndarray, scale: int = 6) -> str:
    # Upscale with nearest-neighbour so each of the 28x28 pixels stays a crisp block.
    size = mnist_uint8.shape[0] * scale
    buf = io.BytesIO()
    Image.fromarray(mnist_uint8).resize((size, size), Image.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- result panel
def result_html(name: str, probs, compact: bool = False) -> str:
    size_cls = " small" if compact else ""
    title = f'{name}<span class="acc">{MODEL_INFO[name]["acc"]} on the MNIST test set</span>'
    if probs is None:
        return f"""
<div class="sheet{size_cls}"><h3>{title}</h3>
  <div class="readout"><div class="digit empty{size_cls}">?</div>
    <div class="verdict"><div class="note">Draw a digit on the black slate and press Predict, or upload a photo.</div></div>
  </div>
</div>"""

    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    conf = float(probs[top])
    if conf >= LOW_CONFIDENCE:
        note = "Confident read." if conf >= 0.95 else f"Second guess: {second} ({probs[second]:.0%})."
    else:
        note = f"Not sure. It could also be {second}. Try drawing larger, in one clear shape."

    bars = "".join(
        f'<div class="bar-row{" top" if d == top else ""}">'
        f'<span class="d">{d}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%"></div></div>'
        f'<span class="v">{p * 100:.1f}%</span></div>'
        for d, p in enumerate(probs)
    )
    return f"""
<div class="sheet{size_cls}"><h3>{title}</h3>
  <div class="readout">
    <div class="digit{size_cls}" aria-label="Predicted digit {top}">{top}</div>
    <div class="verdict"><div class="conf">{conf:.1%}</div><div class="note">{note}</div></div>
  </div>
  <div class="bars" role="list" aria-label="Probability for each digit">{bars}</div>
</div>"""


# ---------------------------------------------------------------- page
html(
    """
<div class="hero">
  <h1>Digits Classifier</h1>
  <p>Draw any digit from 0 to 9. Press Predict and a convolutional network trained on MNIST reads it, showing how sure it is about each digit.</p>
</div>
"""
)
st.write("")

available = [name for name, info in MODEL_INFO.items() if info["path"].exists()]
if "CNN" not in available:
    st.error(
        "Model weights not found at `weights/best_mnist_cnn_model.pth`. "
        "Copy the file from your Google Drive folder into the `weights/` folder and redeploy."
    )
    st.stop()

mode_options = ["CNN"] + (["MLP", "Compare both"] if "MLP" in available else [])
mode = "CNN"
if len(mode_options) > 1:
    mode = st.segmented_control(
        "Model", mode_options, default="CNN", label_visibility="collapsed"
    ) or "CNN"

left, right = st.columns([1, 1.15], gap="large")

# The last predicted drawing survives the canvas reset, so the result stays
# on screen while the slate is empty and ready for the next digit.
if "slate_id" not in st.session_state:
    st.session_state.slate_id = 0
if "last_drawing" not in st.session_state:
    st.session_state.last_drawing = None

mnist_img = None
with left:
    draw_tab, upload_tab = st.tabs(["Draw", "Upload a photo"])

    with draw_tab:
        html('<div style="height:1.6rem" aria-hidden="true"></div>')  # room for the canvas toolbar
        canvas = st_canvas(
            stroke_width=STROKE_WIDTH,
            stroke_color="#FFFFFF",       # white ink ...
            background_color="#000000",   # ... on black, exactly like MNIST
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode="freedraw",
            update_streamlit=True,        # keep Python in sync after every stroke
            return_image_data=True,
            # A new key mounts a fresh, empty canvas: this is how it gets cleared.
            key=f"slate_{st.session_state.slate_id}",
        )

        predict_col, clear_col = st.columns([1.3, 1])
        predict_clicked = predict_col.button(
            "Predict", type="primary", icon=":material/check:", use_container_width=True
        )
        clear_clicked = clear_col.button(
            "Clear", icon=":material/restart_alt:", use_container_width=True
        )
        html(
            '<p class="hint">Draw one digit, filling most of the slate, then press Predict. '
            "The slate clears itself so you can draw the next one.</p>"
        )

        if predict_clicked:
            drawn = (
                to_mnist_layout(canvas_to_gray(canvas.image_data))
                if canvas.image_data is not None
                else None
            )
            if drawn is None:
                st.warning("The slate is empty. Draw a digit first.")
            else:
                st.session_state.last_drawing = drawn
                st.session_state.slate_id += 1   # clear the slate
                st.rerun()

        if clear_clicked:
            st.session_state.slate_id += 1
            st.rerun()

        mnist_img = st.session_state.last_drawing

    with upload_tab:
        upload = st.file_uploader(
            "Photo of a single handwritten digit", type=["png", "jpg", "jpeg", "webp"]
        )
        if upload is not None:
            photo = Image.open(upload)
            st.image(photo, caption="Your photo", width=CANVAS_SIZE)
            photo_mnist = to_mnist_layout(photo_to_gray(photo))
            if photo_mnist is None:
                st.warning("No digit found in this photo. Try a darker pen on plain paper, cropped close.")
            else:
                mnist_img = photo_mnist  # an uploaded photo takes priority over the drawing
                html(
                    '<p class="hint">Showing the prediction for the photo. '
                    "Remove it to go back to drawing.</p>"
                )

    if mnist_img is not None:
        html(
            f"""
<div class="seen">
  <img src="{png_data_uri(mnist_img)}" width="84" height="84" alt="28 by 28 pixel image the model receives">
  <p>What the model received for this prediction: your digit cropped, scaled into a 20px box and centered in a 28×28 frame, the same layout as MNIST.</p>
</div>"""
        )

with right:
    names = ["CNN", "MLP"] if mode == "Compare both" else [mode]
    cols = st.columns(len(names)) if len(names) > 1 else [st.container()]
    for col, name in zip(cols, names):
        probs = predict(get_model(name), mnist_img) if mnist_img is not None else None
        with col:
            html(result_html(name, probs, compact=len(names) > 1))

st.write("")
with st.expander("How the prediction is made"):
    st.markdown(
        """
1. **Capture.** The slate is 280×280 pixels with white ink on black, the same colors as MNIST.
2. **Crop and scale.** The drawing is cropped to its ink, then resized so its longer side is 20 pixels, keeping its proportions.
3. **Center.** It is placed in a 28×28 frame and shifted so its center of mass sits in the middle, which is how MNIST digits were prepared.
4. **Normalize.** Pixels are scaled to 0–1 and then normalized with mean 0.5 and standard deviation 0.5, identical to `transforms.Normalize((0.5,), (0.5,))` used in training.
5. **Predict.** The model runs in evaluation mode and a softmax turns its outputs into the probabilities shown.

**CNN architecture.** Four 3×3 convolution blocks (4, 8, 16, 32 channels) with batch normalization and ReLU, two max-pooling layers, then a dense head of 128 and 64 units with batch normalization and dropout of 0.5. It reached 99.33% accuracy on the 10,000 MNIST test images, compared with 97.56% for the MLP baseline.
"""
    )

st.markdown(
    '<p class="foot">Built by Bertin Iradukunda. Code and training notebook on '
    '<a href="https://github.com/Bertin-Ir/Digits_Classifier">GitHub</a>.</p>',
    unsafe_allow_html=True,
)