import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import numpy as np
import tempfile
import os
import traceback
from PIL import Image
import base64
from io import BytesIO
import json
import time
from datetime import datetime
import cv2  # OpenCV import


# ---------- Helpers ----------
@st.cache_data
def img_to_base64(path, max_width=None):
    if not path or not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    if max_width and img.width > max_width:
        wpercent = (max_width / float(img.width))
        hsize = int((float(img.height) * float(wpercent)))
        img = img.resize((max_width, hsize), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b}"


@st.cache_resource
def load_model():
    """
    Loads the Keras model and also finds and caches the
    base_model and last_conv_layer_name for Grad-CAM.
    """
    MODEL_PATH = "v50.keras"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # --- Logic for Grad-CAM ---
    try:
        base_model = model.get_layer('efficientnetb0')
    except Exception as e:
        st.error(f"Could not find 'efficientnetb0' layer in model: {e}")
        return model, None, None

    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        try:
            # Use layer.output.shape instead of layer.output_shape
            if len(layer.output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            continue

    # Fallback if logic fails
    if last_conv_layer_name is None:
        fallback_layers = ['top_conv', 'block7a_project_conv', 'block6e_project_conv']
        for ln in fallback_layers:
            try:
                if base_model.get_layer(ln):
                    last_conv_layer_name = ln
                    break
            except Exception:
                pass

    if last_conv_layer_name is None:
        st.error("Failed to find a suitable last convolutional layer for Grad-CAM.")

    print(f"Model loaded. Base: {base_model.name}. Conv layer: {last_conv_layer_name}")
    return model, base_model, last_conv_layer_name


def model_prediction(img_path):
    """
    Runs model prediction.
    Returns:
        probs (dict): Dictionary of class probabilities.
        arr (np.array): The preprocessed image array (for Grad-CAM).
    """
    model, _, _ = load_model()
    IMG_SIZE = (224, 224)
    pil = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil).astype(np.float32)
    arr_preprocessed = eff_preprocess(arr)
    arr_expanded = np.expand_dims(arr_preprocessed, axis=0)

    preds = model.predict(arr_expanded)
    preds = np.array(preds).squeeze()

    if not np.all((preds >= 0) & (preds <= 1)):
        probs_array = tf.nn.softmax(preds).numpy()
    else:
        probs_array = preds

    probs_dict = {CLASS_NAMES[i]: float(probs_array[i]) for i in range(len(CLASS_NAMES))}

    return probs_dict, arr_expanded


# --- Grad-CAM Helper Functions ---

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates the raw heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)  # Add epsilon
    return heatmap.numpy()


def generate_gradcam_overlay(original_img_path, preprocessed_img_array, base_model, last_conv_layer_name):
    """Applies all CV2 processing to create the final superimposed image."""

    # 1. Generate Raw Heatmap
    heatmap = make_gradcam_heatmap(preprocessed_img_array, base_model, last_conv_layer_name)

    # 2. Load Original Image (not preprocessed)
    # Using cv2.imread to keep it in cv2 format
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB

    # 3. Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 4. Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), sigmaX=0)

    # 5. Re-normalize after blur
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)  # Add epsilon

    # 6. Apply inverted TURBO colormap
    heatmap_color = np.uint8(255 * (1.0 - heatmap))  # invert
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_TURBO)

    # 7. Apply circular mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    h, w = mask.shape
    center = (w // 2, h // 2)
    radius = min(h // 2, w // 2) - 10  # Robust radius
    cv2.circle(mask, center, radius, 255, -1)

    heatmap_color_masked = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)
    img_masked = cv2.bitwise_and(img.astype("uint8"), img.astype("uint8"), mask=mask)

    # 8. Blend
    alpha = 0.35
    superimposed_img = cv2.addWeighted(heatmap_color_masked, alpha, img_masked, 1 - alpha, 0)

    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    return superimposed_img


# ---------- Config & content ----------
st.set_page_config(page_title="VisionxAid", layout="wide")
MODEL_PATH = "v50.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['AMD', 'DR', 'Glaucoma', 'Normal']

CLASS_IMAGES = {
    "AMD": "assets/amd.jpg",
    "DR": "assets/dr.jpg",
    "Glaucoma": "assets/glaucoma.jpg",
    "Normal": "assets/normal.jpg"
}

CARD_IMAGE_MAX_WIDTH = 520
IMAGE_DISPLAY_WIDTH = 360
BANNER_IMAGE = "assets/banner.jpg"

DISEASE_INFO = {
    "AMD": {
        "title": "Age-related Macular Degeneration (AMD)",
        "brief": "Degeneration of the macula — central vision loss or distortion.",
        "key_points": [
            "Typically affects older adults (50+).",
            "Symptoms: blurred central vision, metamorphopsia.",
            "Management: monitoring, anti-VEGF for wet AMD."
        ]
    },
    "DR": {
        "title": "Diabetic Retinopathy (DR)",
        "brief": "Microvascular retinal damage from diabetes — can progress to vision loss.",
        "key_points": [
            "Related to long-standing or poorly-controlled diabetes.",
            "Signs: microaneurysms, hemorrhages, exudates.",
            "Management: screening, glycemic control, laser/anti-VEGF."
        ]
    },
    "Glaucoma": {
        "title": "Glaucoma",
        "brief": "Optic nerve damage, often with raised IOP — early peripheral field loss.",
        "key_points": [
            "Often asymptomatic early.",
            "Diagnosis: IOP, optic disc, visual fields.",
            "Management: drops, laser, or surgery to lower IOP."
        ]
    }
}

# ---------- Global CSS (page-level) ----------
st.markdown(
    """
    <style>
    /* center main content and constrain width */
    .app-container { max-width:1100px; margin-left:auto; margin-right:auto; padding-top:0px; }

    /* feature tiles and simple typography */
    .feature-row { display:flex; gap:16px; justify-content:center; margin-top:6px; margin-bottom:6px; }
    .feature-tile { width:160px; height:160px; background:rgba(61, 157, 243, 0.2);opacity:1; border-radius:10px; display:flex; align-items:center; justify-content:center;  box-shadow:0 8px 22px rgba(0,0,0,0.08);font-weight:700; font-size:1rem;}

    .muted { color:#6b7280; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Predict", "Recommendations", "About"])

# ---------- Header / Banner ----------

# Convert banner to base64
banner_b64 = img_to_base64(BANNER_IMAGE)

# Display banner using BASE64
if banner_b64:
    st.markdown(
        f"""
        <div style='display:flex; justify-content:center; margin-top:0px;margin-bottom:12px;'>
            <img src='{banner_b64}' style='width:100%; border-radius:3px;border-top:0px;padding:0px;'/>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Banner image not found or failed to load.")

st.markdown('<div class="app-container">', unsafe_allow_html=True)
# ---------------- HOME ----------------
if page == "Home":
    # Quick overview
    st.markdown("### Quick overview")
    st.markdown(
        "<div class='muted'><strong>Model:</strong> EfficientNet (v50.keras)  •  <strong>Input:</strong> Fundus (retinal) images (RGB JPEG/PNG)  •  <strong>Task:</strong> Image classification (AMD / DR / Glaucoma / Normal)</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Build combined card HTML (single string) and render with components.html
    card_htmls = []
    debug_msgs = []
    for key in ["AMD", "DR", "Glaucoma"]:
        info = DISEASE_INFO.get(key, {})
        title = info.get("title", key)
        brief = info.get("brief", "")
        bullets = info.get("key_points", [])
        bullets_html = "".join([f"<li>{b}</li>" for b in bullets])

        img_path = CLASS_IMAGES.get(key)
        b64 = None
        try:
            b64 = img_to_base64(img_path, max_width=CARD_IMAGE_MAX_WIDTH) if img_path else None
            if b64 is None:
                debug_msgs.append(f"image missing: {img_path}")
        except Exception as e:
            b64 = None
            debug_msgs.append(f"image conversion error for {img_path}: {e}")

        if b64:
            media = f'<div class="media"><img src="{b64}" alt="{key}"></div>'
        else:
            media = f'<div class="media" style="font-weight:700;color:#111827;font-size:1.05rem;display:flex;align-items:center;justify-content:center;height:160px;">{title}</div>'

        card_html = f"""
        <div class="card">
          {media}
          <div class="title">{title}</div>
          <div class="brief">{brief}</div>
          <ul>{bullets_html}</ul>
        </div>
        """
        card_htmls.append(card_html)

    combined_cards = '<div class="card-row">' + "".join(card_htmls) + '</div>'

    # If images had problems show a small red debug notice
    if debug_msgs:
        dbg_html = "<div style='color:#b91c1c;margin-bottom:6px;'><strong>Image issues:</strong><ul>"
        dbg_html += "".join([f"<li>{m}</li>" for m in debug_msgs])
        dbg_html += "</ul></div>"
        st.markdown(dbg_html, unsafe_allow_html=True)

    # Render the cards as real DOM via components.html (prevents Streamlit escaping)
    components.html(
        f"""
        <div style="padding:6px 0">{combined_cards}</div>
        <style>
        .card-row {{ display:flex; gap:20px; justify-content:center; align-items:stretch; flex-wrap:nowrap; overflow-x:auto; padding-bottom:8px; }}
        .card {{ width:320px; min-height:360px; box-sizing:border-box; padding:16px; border-radius:10px; background:rgba(61, 157, 243, 0.08); opacity:2; display:flex; flex-direction:column; margin:0 6px; transition: all .18s ease;font-family: "Inter", sans-serif; }}
        .media {{ width:100%; height:170px; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center; margin-bottom:12px; }}
        .media img {{ width:100%; height:100%; object-fit:contain; display:block;background:black; }}
        .title {{ font-size:1.1rem; font-weight:700; margin-bottom:10px; color:#fff;line-height:1.3; }}
        .brief {{ color:#fff; margin-bottom:12px; font-size:0.95rem; }}
        .card ul {{ padding-left:18px; margin-top:6px; font-size:0.92rem; color:#fff;line-height:1.4; }}
        .card:hover {{ transform: translateY(-6px); box-shadow:0 10px 28px rgba(0,0,0,0.12); }}
        </style>
        """,
        height=480,
        scrolling=True,

    )

    st.markdown("---")

    # Features (small tiles)
    feature_html = """
    <style>
    .feature-row { display:flex; gap:16px; justify-content:center; margin-top:6px; margin-bottom:6px; }
    .feature-tile { width:160px; height:160px; background:rgba(61, 157, 243, 0.08); opacity:1; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow: 0 4px 12px rgba(0,0,0,0.04); font-weight:700; color:#fff; font-size:1rem; }
    .feature-tile:hover {  transform: translateY(-6px); box-shadow:0 10px 28px rgba(0,0,0,0.12);}
    </style>
    <div class="feature-row">
      <div class="feature-tile">Prediction</div>
      <div class="feature-tile">Recommendations</div>
      <div class="feature-tile">Uploads</div>
    </div>
    """
    st.markdown("### Features")
    st.markdown(feature_html, unsafe_allow_html=True)

    st.markdown("---")

    # How-to
    st.markdown("### How to use the Tool?")
    st.markdown(
        """
        1. Go to Predict in the left navigation.  
        2. Upload a fundus image (JPEG/PNG).  
        3. Click Predict and review confidence scores and the Grad-CAM heatmap.
        4. Optionally click Save prediction to uploads/ to store a JSON report in uploads/.  
        5. View saved reports in Recent uploads.
        """
    )

    st.markdown("---")

    # Recent uploads + Stats side-by-side (robust)
    left_col, right_col = st.columns([2, 1])
    uploads_dir = "uploads"

    # LEFT: Recent uploads (files sorted by mtime, newest first)
    with left_col:
        st.markdown("### Recent uploads")
        if not os.path.exists(uploads_dir):
            st.info("No uploads yet. Save a prediction to populate this list.")
        else:
            files = sorted(
                [(os.path.getmtime(os.path.join(uploads_dir, f)), f) for f in os.listdir(uploads_dir)],
                key=lambda x: x[0],
                reverse=True,
            )
            if not files:
                st.info("No uploads found.")
            else:
                # show up to 12 recent items
                for ts, fname in files[:12]:
                    path = os.path.join(uploads_dir, fname)
                    pretty_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    with st.expander(f"{fname} — {pretty_time}"):
                        lower = fname.lower()
                        # images
                        if lower.endswith((".jpg", ".jpeg", ".png")):
                            try:
                                # *** Use use_container_width ***
                                st.image(path, use_container_width=True)
                            except Exception as e:
                                st.write("Unable to show image:", e)
                        # jsons
                        elif lower.endswith(".json"):
                            try:
                                with open(path, "r") as fh:
                                    data = json.load(fh)
                                # show compact summary first
                                pred = data.get("prediction") or data.get("top_label") or data.get("pred") or "N/A"
                                top_conf = data.get("top_confidence") or data.get("top_conf") or None

                                # if top_conf missing, try to compute from probs
                                if top_conf is None:
                                    probs = data.get("probs")
                                    if isinstance(probs, dict):
                                        try:
                                            top_conf = max(probs.values())
                                        except Exception:
                                            top_conf = None
                                    elif isinstance(probs, (list, tuple)):
                                        try:
                                            top_conf = max(probs)
                                        except Exception:
                                            top_conf = None

                                summary = {
                                    "file": data.get("file", "unknown"),
                                    "prediction": pred,
                                    "top_confidence": top_conf,
                                    "saved_at": data.get("saved_at")
                                }
                                st.json(summary)

                                # show full json collapsed below
                                with st.expander("Full JSON"):
                                    st.json(data)

                                # try to display any related image(s) with similar base name
                                base = os.path.splitext(fname)[0]
                                related = [f for f in os.listdir(uploads_dir) if
                                           base in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
                                for r in related[:3]:
                                    rpath = os.path.join(uploads_dir, r)
                                    try:
                                        # *** Use use_container_width ***
                                        st.image(rpath, caption=f"Related file: {r}", use_container_width=True)
                                    except Exception:
                                        pass

                            except Exception as e:
                                st.write("Unable to open JSON:", e)
                        else:
                            st.write("File:", fname)

    # RIGHT: Stats / Accuracy (from metrics.json if present, otherwise compute from saved JSONs)
    with right_col:
        st.markdown("### Stats / Accuracy")
        metrics_file = os.path.join(uploads_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as fh:
                    metrics = json.load(fh)
                reported_acc = metrics.get("accuracy")
                st.markdown("Reported model metrics (from uploads/metrics.json)")
                if reported_acc is not None:
                    st.metric("Accuracy", f"{reported_acc * 100:.2f}%")
                for k, v in metrics.items():
                    if k != "accuracy":
                        st.write(f"- {k}: {v}")
            except Exception as e:
                st.write("Failed to read metrics.json:", e)
        else:
            # scan uploaded JSONs and compute stats
            confs = []  # top confidences (floats 0..1)
            pred_counts = {}  # counts per predicted label
            json_count = 0
            if os.path.exists(uploads_dir):
                for fname in os.listdir(uploads_dir):
                    if not fname.lower().endswith(".json"):
                        continue
                    path = os.path.join(uploads_dir, fname)
                    try:
                        with open(path, "r") as fh:
                            data = json.load(fh)
                        json_count += 1

                        # preferred direct fields
                        top_conf = None
                        if "top_confidence" in data:
                            try:
                                top_conf = float(data["top_confidence"])
                            except Exception:
                                top_conf = None
                        elif "top_conf" in data:
                            try:
                                top_conf = float(data["top_conf"])
                            except Exception:
                                top_conf = None

                        # fallback: compute from probs (dict or list)
                        if top_conf is None:
                            probs = data.get("probs")
                            if isinstance(probs, dict):
                                try:
                                    top_conf = max([float(v) for v in probs.values()])
                                except Exception:
                                    top_conf = None
                            elif isinstance(probs, (list, tuple)):
                                try:
                                    top_conf = max([float(v) for v in probs])
                                except Exception:
                                    top_conf = None

                        if top_conf is not None:
                            confs.append(max(0.0, min(1.0, float(top_conf))))  # clamp 0..1

                        # count predicted label if available
                        label = data.get("prediction") or data.get("top_label") or data.get("pred")
                        if label:
                            pred_counts[label] = pred_counts.get(label, 0) + 1

                    except Exception:
                        # skip malformed JSON
                        continue

            total_upload_files = len(os.listdir(uploads_dir)) if os.path.exists(uploads_dir) else 0
            st.metric("Total files in uploads/", total_upload_files)
            st.metric("JSON reports", json_count)

            if confs:
                avg_conf = sum(confs) / len(confs)
                st.metric("Avg top confidence", f"{avg_conf * 100:.2f}%")
                st.write(f"- Computed from {len(confs)} saved JSON(s).")
            else:
                st.info("No saved prediction JSONs to compute stats. Save predictions to uploads/ first.")

            # show counts per prediction label (if any)
            if pred_counts == 0:
                # optionally remind how to produce stats
                if json_count:
                    st.write("- No prediction labels found in JSONs.")
                else:
                    st.write("- No JSON prediction reports found.")
    st.markdown("---")
    st.markdown("Made with ❤ — Model: v50.keras.")

# ------------------ PREDICT (with Grad-CAM) ------------------
# ------------------ PREDICT (with Grad-CAM) ------------------
elif page == "Predict":
    import os
    import time
    import json
    import tempfile
    import traceback
    from datetime import datetime
    import numpy as np
    import plotly.graph_objects as go
    # cv2 is imported globally

    # Imports for PDF Report Generation
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch  # Added for easier margin setting

    st.header("Predict — Upload Image")
    st.write("Supported formats: JPG / PNG. Recommended image size: > 224×224 px.")

    # Uploader
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    img_path = None

    # Clear session state if no file is uploaded
    if uploaded is None:
        st.session_state['probs'] = None
        st.session_state['top_label'] = None
        st.session_state['top_conf'] = None
        st.session_state['heatmap_img'] = None
        st.session_state['img_path'] = None  # Clear image path

    if uploaded is not None:
        # Save temporary file for inference
        suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            img_path = tmp.name
            st.session_state['img_path'] = img_path  # Store for PDF report

        # TOP ROW: image (left) + text/actions (right)
        left, right = st.columns([0.7, 1.3])
        with left:
            st.markdown("Uploaded image")
            st.image(img_path, width=IMAGE_DISPLAY_WIDTH)

        with right:
            st.markdown("Prediction & Actions")

            if st.button("Predict"):
                with st.spinner("Analyzing image and generating heatmap..."):
                    try:
                        # 1. Get Prediction
                        probs, preprocessed_arr = model_prediction(img_path)
                        top_label = max(probs, key=probs.get)
                        top_conf = probs[top_label]

                        # 2. Get Model components for Grad-CAM
                        model, base_model, last_conv_layer_name = load_model()

                        # 3. Generate Heatmap
                        if base_model and last_conv_layer_name:
                            heatmap_img = generate_gradcam_overlay(
                                img_path,
                                preprocessed_arr,
                                base_model,
                                last_conv_layer_name
                            )
                            st.session_state['heatmap_img'] = heatmap_img
                        else:
                            st.session_state['heatmap_img'] = None
                            st.warning("Could not generate heatmap (model layer issue).")

                        # 4. Store results in session state
                        st.session_state['probs'] = probs
                        st.session_state['top_label'] = top_label
                        st.session_state['top_conf'] = top_conf

                        st.success(f"Prediction: {top_label} — {top_conf * 100:.2f}%")

                    except Exception:
                        st.error("Prediction failed. See traceback below.")
                        st.text(traceback.format_exc())
                        st.session_state['probs'] = None
                        st.session_state['heatmap_img'] = None

            # Show probabilities list if available from session state
            current_probs = st.session_state.get('probs')
            if current_probs is not None:
                st.markdown("Probabilities")
                for lab in CLASS_NAMES:
                    val = current_probs.get(lab, 0.0)
                    st.write(f"- {lab}: {val * 100:.2f}%")
            else:
                st.info("No prediction yet. Click Predict to compute probabilities.")

            st.markdown("")

            # -------------------- PDF REPORT BUTTON (REVISED) --------------------
            if st.session_state.get("top_label") is not None:

                if st.button("Generate Report"):
                    try:
                        os.makedirs("reports", exist_ok=True)
                        pdf_path = f"reports/report_{int(time.time())}.pdf"

                        # Get data for report
                        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        original_filename = uploaded.name if uploaded else "N/A"
                        pred_label = st.session_state['top_label']
                        pred_conf = st.session_state['top_conf'] * 100
                        original_img_for_report = st.session_state.get('img_path')

                        # Create PDF
                        c = canvas.Canvas(pdf_path, pagesize=letter)
                        width, height = letter
                        margin = 0.75 * inch

                        # --- 1. Header ---
                        c.setFont("Helvetica-Bold", 20)
                        c.drawString(margin, height - margin, "VisionxAid - Retinal Analysis Report")

                        c.setFont("Helvetica", 10)
                        c.drawString(margin, height - margin - 20, f"Report Generated: {report_time}")
                        c.line(margin, height - margin - 25, width - margin, height - margin - 25)

                        # --- 2. Summary (Two Columns) ---
                        y_pos = height - margin - 60

                        # Column 1: File Info
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(margin, y_pos, "File Information")
                        c.setFont("Helvetica", 11)
                        c.drawString(margin + 10, y_pos - 18, f"Original File: {original_filename}")

                        # Column 2: Prediction
                        col_2_x = margin + 300
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(col_2_x, y_pos, "Prediction Summary")
                        c.setFont("Helvetica", 11)
                        c.drawString(col_2_x + 10, y_pos - 18, f"Predicted Condition: {pred_label}")
                        c.drawString(col_2_x + 10, y_pos - 36, f"Confidence: {pred_conf:.2f}%")

                        y_pos -= 70
                        c.line(margin, y_pos, width - margin, y_pos)

                        # --- 3. Recommendations ---
                        y_pos -= 30
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(margin, y_pos, "Recommendations")
                        y_pos -= 25
                        c.setFont("Helvetica", 11)

                        if pred_label == "AMD":
                            recs = [
                                "• Schedule regular retinal checkups.",
                                "• Increase intake of leafy green vegetables.",
                                "• Monitor any sudden central vision distortion."
                            ]
                        elif pred_label == "DR":
                            recs = [
                                "• Maintain strict blood sugar control.",
                                "• Get a diabetic retinal exam annually.",
                                "• Seek medical help if vision becomes blurry."
                            ]
                        elif pred_label == "Glaucoma":
                            recs = [
                                "• Monitor intraocular pressure routinely.",
                                "• Use prescribed anti-glaucoma eye drops.",
                                "• Follow-up every 3 to 6 months."
                            ]
                        else:
                            recs = ["• Normal retina. Maintain regular eye check-ups."]

                        for r in recs:
                            c.drawString(margin + 10, y_pos, r)
                            y_pos -= 20

                        y_pos -= 20
                        c.line(margin, y_pos, width - margin, y_pos)

                        # --- 4. Visual Analysis (Side-by-Side) ---
                        y_pos -= 30
                        c.setFont("Helvetica-Bold", 14)
                        c.drawString(margin, y_pos, "Visual Analysis")

                        img_y = y_pos - 240  # Y-position for top of images
                        img_height = 220
                        img_width = 250
                        gap = 30

                        x_original = margin + 10
                        x_heatmap = x_original + img_width + gap

                        # --- Draw Original Image ---
                        c.setFont("Helvetica-Bold", 11)
                        c.drawCentredString(x_original + img_width / 2, img_y + img_height + 10, "Original Image")
                        if original_img_for_report and os.path.exists(original_img_for_report):
                            try:
                                c.drawImage(ImageReader(original_img_for_report), x_original, img_y,
                                            width=img_width, height=img_height,
                                            preserveAspectRatio=True, anchor='n')
                            except Exception as e:
                                c.setFont("Helvetica", 9)
                                c.drawCentredString(x_original + img_width / 2, img_y + img_height / 2,
                                                    "Failed to load original image.")
                        else:
                            c.setFont("Helvetica", 9)
                            c.drawCentredString(x_original + img_width / 2, img_y + img_height / 2,
                                                "Original image not available.")

                        # --- Draw Heatmap Image ---
                        c.setFont("Helvetica-Bold", 11)
                        c.drawCentredString(x_heatmap + img_width / 2, img_y + img_height + 10, "Grad-CAM Heatmap")
                        if "heatmap_img" in st.session_state and st.session_state['heatmap_img'] is not None:
                            try:
                                heat_path = "reports/heatmap_temp.jpg"
                                # Convert RGB numpy array (from session) to BGR for cv2.imwrite
                                heatmap_bgr = cv2.cvtColor(st.session_state["heatmap_img"], cv2.COLOR_RGB2BGR)
                                cv2.imwrite(heat_path, heatmap_bgr)

                                c.drawImage(ImageReader(heat_path), x_heatmap, img_y,
                                            width=img_width, height=img_height,
                                            preserveAspectRatio=True, anchor='n')
                            except Exception as e:
                                c.setFont("Helvetica", 9)
                                c.drawCentredString(x_heatmap + img_width / 2, img_y + img_height / 2,
                                                    "Failed to load heatmap.")
                        else:
                            c.setFont("Helvetica", 9)
                            c.drawCentredString(x_heatmap + img_width / 2, img_y + img_height / 2,
                                                "Heatmap not available.")

                        # --- 5. Footer ---
                        c.setFont("Helvetica-Oblique", 9)
                        c.line(margin, margin - 15, width - margin, margin - 15)
                        c.drawCentredString(width / 2, margin - 30,
                                            "Disclaimer: This report is generated by an AI model and is for informational purposes only. Not for clinical diagnosis.")

                        # Save PDF
                        c.save()

                        # Show success + download button
                        st.success("report generated successfully!")

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download PDF Report",
                                data=f,
                                file_name=f"VisionxAid_Report_{original_filename}.pdf",
                                mime="application/pdf"
                            )

                    except Exception as e:
                        st.error("Failed to generate report.")
                        st.text(traceback.format_exc())

            # -------------------- END PDF REPORT BUTTON --------------------

            # Save JSON action
            if st.button("Save prediction to uploads"):
                # Use session state probs if available
                save_probs = st.session_state.get('probs')
                # Use session state img_path if available
                save_img_path = st.session_state.get('img_path')

                try:
                    os.makedirs("uploads", exist_ok=True)

                    # If probs aren't in state, run prediction first
                    if save_probs is None:
                        if save_img_path is None:
                            st.error("No image available to predict/save.")
                            # Use 'st.stop()' to halt execution in this callback
                            st.stop()
                        else:
                            with st.spinner("Computing prediction before save..."):
                                save_probs, _ = model_prediction(save_img_path)  # Don't need heatmap

                    top_label = max(save_probs, key=save_probs.get)
                    top_conf = save_probs[top_label]

                    fname = os.path.join("uploads", f"pred_{int(time.time())}.json")
                    pred_data = {
                        "file": uploaded.name,
                        "prediction": top_label,
                        "top_confidence": top_conf,
                        "probs": save_probs,
                        "saved_at": datetime.utcnow().isoformat() + "Z"
                    }
                    with open(fname, "w") as f:
                        json.dump(pred_data, f, indent=2)

                    st.success(f"Saved report to {fname}")

                except Exception as e:
                    st.error(f"Save failed: {e}")
                    st.text(traceback.format_exc())

        # ----------------- BOTTOM SECTION: Confidence Distribution & Grad-CAM -----------------
        st.markdown("---")
        st.markdown("### Prediction Analysis")

        # Read results from session state
        probs = st.session_state.get('probs')
        top_label = st.session_state.get('top_label')
        top_conf = st.session_state.get('top_conf')
        heatmap_img = st.session_state.get('heatmap_img')

        if probs is None:
            st.info("Click 'Predict' to see analysis charts.")
        else:
            # Prepare data for pie chart
            pct = (np.array([probs[l] for l in CLASS_NAMES]) * 100).tolist()
            palette = ['#4F81BD', '#C0504D', '#9BBB59', '#8064A2']
            colors = palette[:len(pct)]

            fig_pie = go.Figure(data=[go.Pie(
                labels=CLASS_NAMES,
                values=pct,
                hole=0.42,
                sort=False,
                marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.55)', width=3)),
                textinfo='percent',
                hoverinfo='label+percent',
                textfont=dict(color='white', size=12, family='Inter, sans-serif')
            )])

            fig_pie.update_layout(
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font=dict(color='white', size=15, family='Inter, sans-serif'),
                height=387,
                autosize=True,
                legend=dict(
                    orientation="v",
                    bgcolor='rgba(28,31,38,0.6)',
                    bordercolor='rgba(255,255,255,0.06)',
                    font=dict(color='white')
                ),
                margin=dict(l=5, r=5, t=7, b=5),
                showlegend=True
            )

            fig_pie.add_annotation(
                x=0.5, y=0.5,
                text=f"<b>{top_label}</b><br>{top_conf * 100:.1f}%",
                showarrow=False,
                font=dict(size=13, color='white', family='Inter, sans-serif'),
                align="center",
                xref="paper", yref="paper",
            )

            # --- Display Charts Side-by-Side ---
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown("##### Prediction Probabilities")
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with col2:
                st.markdown("##### Grad-CAM Heatmap")
                if heatmap_img is not None:
                    st.image(
                        heatmap_img,
                        caption=f"Model focus for '{top_label}' prediction",
                        use_container_width=True
                    )
                else:
                    st.info("Heatmap could not be generated.")

    else:
        st.info("Upload a fundus image (JPG/PNG) to run prediction.")
    st.markdown("---")
    st.markdown("Made with ❤ — Model: v50.keras.")
# ------------------------------------------------------------------------

# -------------- RECOMMENDATIONS --------------
elif page == "Recommendations":
    st.header("Recommendations & Next Steps (non-diagnostic)")
    st.info("This section provides general guidance and resources — NOT a substitute for clinical evaluation.")
    st.markdown("""
    If the model predicts any abnormal class (e.g., AMD, DR, Glaucoma):
    - Advise the patient to seek ophthalmic evaluation as soon as possible.
    - Share the image and model result with a retina specialist for confirmation.
    - Consider additional tests such as OCT-Angiography, fluorescein angiography or visual field testing (for glaucoma).

    General patient guidance
    - Maintain blood sugar control (for diabetic retinopathy).
    - Manage blood pressure and cardiovascular risk factors.
    - Attend regular ophthalmic follow-ups as recommended.
    """)
    st.markdown("---")
    st.markdown("Made with ❤ — Model: v50.keras.")

# ------------------ ABOUT ------------------
elif page == "About":
    import os

    st.header("About VisionxAid")
    st.markdown(
        "<div class='muted'>A research/educational project that uses an EfficientNet-based CNN to classify common retinal conditions from fundus images. "
        "This app is <strong>NOT</strong> a diagnostic tool — results are for research/educational use only.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # TEAM ROW
    st.subheader("The Team")

    # Team data
    team = [
        {"name": "Neil Varghese Abraham", "role": "Developer (1BY23CS144)", "img": "assets/neil.jpg"},
        {"name": "Pushpendra Kumar Garg", "role": "Developer (1BY23CS167)", "img": "assets/pushpendra.jpg"},
        {"name": "Sai Mansi Maddali", "role": "Developer (1BY23CS196)", "img": "assets/sai_mansi.jpg"},
        {"name": "Trinabh Chadda", "role": "Developer (1BY23CS275)", "img": "assets/trinabh.jpg"},
    ]

    # Build team card HTML
    team_cards = []
    for member in team:
        # inside the loop that builds team_cards (About page)
        img_path = member.get("img")
        b64 = None
        try:
            # convert to base64 (use a modest max width so embedded HTML isn't huge)
            b64 = img_to_base64(img_path, max_width=420) if img_path and os.path.exists(img_path) else None
        except Exception:
            b64 = None

        if b64:
            # use data URI so the image is embedded inside the components.html iframe
            media = f"<img src='{b64}' class='team-img'/>"
        else:
            initials = "".join([p[0] for p in member['name'].split()[:2]]).upper()
            media = f"<div class='initials'>{initials}</div>"


        card_html = f"""
        <div class="team-card">
          <div class="team-media">
            {media}
          </div>
          <div class="team-meta">
            <div class="team-name">{member['name']}</div>
            <div class="team-role">{member['role']}</div>
          </div>
        </div>
        """
        team_cards.append(card_html)

    team_row_html = f"""
    <div style="margin-top:10px;">
      <style>
        .team-row {{
          display:flex;
          gap:16px;
          align-items:flex-start;
          overflow-x:auto;
          overflow-y:hidden;
          padding:10px 6px;
          height: 280px;
          white-space: nowrap;
          -webkit-overflow-scrolling: touch;
        }}
        .team-card {{
          display:inline-flex;
          flex-direction:column;
          align-items:center;
          justify-content:flex-start;
          gap:10px;
          min-width: 240px;
          max-width: 240px;
          height: 260px;
          background:rgba(61, 157, 243, 0.08); 
          opacity:1;
          border-radius:12px;
          padding:12px;
          box-sizing:border-box;
          flex: 0 0 auto;
          box-shadow:0 6px 18px rgba(2,6,23,0.04);
          text-align:center;
          color:#fff;
        }}
        .team-media {{ width:100%; display:flex; align-items:center; justify-content:center; }}
        .team-img {{ width:100%; height:140px; border-radius:10px; object-fit:cover; display:block; box-shadow:0 6px 18px rgba(2,6,23,0.06); }}
        .initials {{ width:100%; height:140px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:linear-gradient(90deg,#eef2ff,#fff); font-weight:800; color:#black; font-size:28px; box-shadow:0 6px 18px rgba(2,6,23,0.06); }}
        .team-meta {{ display:flex; flex-direction:column; gap:6px; align-items:center; padding:0 6px; min-width:0; }}
        .team-name {{ font-weight:800; color:#fff; margin:0; font-size:1rem; overflow:hidden; text-overflow:ellipsis; white-space:normal; }}
        .team-role {{ color:#fff; margin:0; font-size:0.9rem; overflow:hidden; text-overflow:ellipsis; white-space:normal; }}
        .team-card:hover {{transform: translateY(-6px);box-shadow: 0 10px 28px rgba(0,0,0,0.12);transition: all .2s ease;}}
      </style>
      <div class="team-row">
        {"".join(team_cards)}
      </div>
    </div>
    """
    components.html(team_row_html, height=300, scrolling=False)

    st.markdown("---")

    # FAQ
    st.subheader("FAQ & Troubleshooting")
    st.markdown("Common questions about the app, inputs, and how to interpret results.")

    with st.expander("What images work best?"):
        st.write(
            "- Use clear fundus (retinal) photos with minimal glare and centered macula/optic disc.\n"
            "- Prefer JPEG/PNG, RGB, resolution at least 224×224 (larger is fine)."
        )

    with st.expander("Is this a diagnostic tool?"):
        st.write(
            "No. This project is for research/education only. Always consult an ophthalmologist for clinical decisions."
        )

    with st.expander("What is the Grad-CAM Heatmap?"):
        st.write(
            "- The heatmap (Grad-CAM) shows the areas of the image the model 'focused on' to make its prediction. \n"
            "- Red/Yellow areas contributed most to the decision. Blue/Purple areas contributed least.\n"
            "- This helps understand why the model chose a specific class."
        )

    with st.expander("How do I save and view predictions?"):
        st.write(
            "- On the Predict page, after making a prediction, click 'Save prediction to uploads/' to store a JSON report in the uploads/ folder.\n"
            "- View saved reports on the Home page under Recent uploads. Each report includes the predicted label and confidence scores."
        )

    with st.expander("Ethics, limitations, and dataset sources"):
        st.write(
            "- The model was trained on a mixture of public and curated datasets — details are documented in project notes.\n"
            "- Limitations: class imbalance, domain shift with imaging devices, and susceptibility to image artifacts.\n"
            "- Intended use: research/education. Not approved for clinical use."
        )

    st.markdown("---")
    st.markdown("Made with ❤ — Model: v50.keras.")