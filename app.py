import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from fpdf import FPDF
import datetime
import io
import tempfile
import torch
from ultralytics.nn.tasks import DetectionModel




# Setup + Model Loading
# --------------------------------------------
# Load model once at the top
@st.cache_resource
def load_model():
    import torch
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

    return YOLO("models/yolov5s.pt")


model = load_model()
CLASSES = model.names

# UI to Select Room + Configure Allowed Classes
# --------------------------------------------
st.title("📦 Location-Based Object Validator")

# Choose a room
rooms = ["Room A", "Rom B", "Conveyor 1", "Conveyor 2"]
selected_room = st.selectbox("Select Room/Conveyor", rooms)

# Initialize rule storage
if "rules" not in st.session_state:
    st.session_state.rules = {}

# Initialize room's object config if missing
if selected_room not in st.session_state.rules:
    st.session_state.rules[selected_room] = {name: True for name in CLASSES.values()}

st.subheader("Allowed Objects in Selected Location")

# Select All / Deselect All Buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("✅ Select All"):
        for name in CLASSES.values():
            st.session_state.rules[selected_room][name] = True
with col2:
    if st.button("🚫 Deselect All"):
        for name in CLASSES.values():
            st.session_state.rules[selected_room][name] = False

# Object checkboxes (grid layout)
allowed_classes = {}
cols = st.columns(4)
for i, name in CLASSES.items():
    with cols[i % 4]:
        allowed = st.checkbox(
            name,
            value=st.session_state.rules[selected_room].get(name, False),
            key=f"{selected_room}_{name}"
        )
        allowed_classes[name] = allowed

# Save updated settings
st.session_state.rules[selected_room] = allowed_classes


# Upload Image & Evaluate
# --------------------------------------------
st.subheader("📤 Upload an image for validation")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Validate file size (between 5 KB and 5 MB)
    file_size_kb = uploaded_file.size / 1024
    if file_size_kb < 5:
        st.error(f"❌ File is too small ({file_size_kb:.1f} KB). Likely not a valid image.")
        st.stop()
    elif file_size_kb > 5120:
        st.error(f"❌ File too large ({file_size_kb:.1f} KB). Must be under 5 MB.")
        st.stop()

    # Validate image content
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a JPG or PNG.")
        st.stop()

    # Run object detection
    results = model(image)[0]
    detected_labels = []
    violations = []

    # Visual feedback container
    st.markdown("## 🧾 Detection Summary")

    if results.boxes is None or len(results.boxes) == 0:
        st.warning("🔍 No objects were detected in the image.")
    else:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = CLASSES[cls_id]
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            detected_labels.append((label, conf))

            if not st.session_state.rules[selected_room].get(label, False):
                violations.append((label, conf))

        if not detected_labels:
            st.warning("🧐 Objects detected, but none passed the confidence threshold (0.5).")
        else:
            for label, conf in detected_labels:
                allowed = (label, conf) not in violations
                bg_color = "#d4edda" if allowed else "#f8d7da"
                text_color = "#155724" if allowed else "#721c24"
                status = "✅ ALLOWED" if allowed else "🚫 NOT ALLOWED"

                st.markdown(f"""
                <div style="padding:8px 12px; border-radius:8px; margin-bottom:6px;
                            background-color:{bg_color}; color:{text_color}; font-weight:600;">
                    {label} ({conf:.2f}) - {status}
                </div>
                """, unsafe_allow_html=True)

            if not violations:
                st.success("🎉 All detected objects are allowed in this location!")
            else:
                st.error(f"🚨 Violations detected: {', '.join(v[0] for v in violations)}")

    
    def sanitize_text(text):
        return text.encode("latin-1", errors="replace").decode("latin-1")

    if detected_labels:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, sanitize_text("Object Detection Report"), ln=True, align="C")
        pdf.ln(5)

        # Room / Conveyor
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, sanitize_text(f"Room / Conveyor: {selected_room}"), ln=True)

        # Allowed Objects
        allowed = [k for k, v in st.session_state.rules[selected_room].items() if v]
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Allowed Objects:", ln=True)
        pdf.set_font("Arial", "", 12)
        for obj in allowed:
            pdf.cell(0, 8, f"- {sanitize_text(obj)}", ln=True)

        # Upload Time
        pdf.ln(2)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, sanitize_text(f"Upload Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)

        # Image Section
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                image.save(tmp_img.name, format="JPEG")
                tmp_img_path = tmp_img.name

            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Uploaded Image:", ln=True)
            pdf.image(tmp_img_path, x=30, w=pdf.w - 60)  # Centered-ish
            pdf.ln(5)

        except Exception as e:
            st.warning(f"Could not embed image: {e}")
        finally:
            if 'tmp_img_path' in locals() and os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)

        # Detected Objects
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detected Objects:", ln=True)
        pdf.set_font("Arial", "", 12)
        for label, conf in detected_labels:
            verdict = "ALLOWED" if (label, conf) not in violations else "NOT ALLOWED"
            pdf.cell(0, 8, sanitize_text(f"- {label} ({conf:.2f}) - {verdict}"), ln=True)

        # Violations
        if violations:
            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Violations Detected:", ln=True)
            pdf.set_font("Arial", "", 12)
            for label, conf in violations:
                pdf.cell(0, 8, sanitize_text(f"- {label} ({conf:.2f})"), ln=True)

        # Export and download
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_output = io.BytesIO(pdf_bytes)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output,
            file_name=f"detection_report_{selected_room}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )