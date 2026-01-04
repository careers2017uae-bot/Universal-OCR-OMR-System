# app.py
"""
Universal OCR + OMR System
Cloud-Native | Jina Vision OCR | Production-Safe
"""

import streamlit as st
from PIL import Image
import requests
from docx import Document
from fpdf import FPDF
import tempfile
import os
import base64

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Universal OCR & OMR System",
    page_icon="📄",
    layout="centered"
)

JINA_API_KEY = st.secrets.get("JINA_API_KEY", "")

JINA_VISION_OCR_ENDPOINT = "https://api.jina.ai/v1/vision/ocr"

# ---------------- UI ----------------

st.title("📄 Universal OCR & OMR System")
st.caption("Multilingual • Handwriting • Cloud-Native AI OCR")

uploaded_file = st.file_uploader(
    "📷 Upload scanned image or camera photo",
    type=["png", "jpg", "jpeg"]
)

output_format = st.radio("📤 Output Format", ["Word (.docx)", "PDF (.pdf)"])
process_btn = st.button("🚀 Process Document")

# ---------------- FUNCTIONS ----------------

def jina_vision_ocr(image: Image.Image) -> str:
    if not JINA_API_KEY:
        st.error("JINA_API_KEY not configured in Streamlit secrets.")
        st.stop()

    buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(buffered.name)

    with open(buffered.name, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(image_bytes).decode()

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "image": encoded
    }

    response = requests.post(
        JINA_VISION_OCR_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()
    return response.json().get("text", "")

def generate_docx(text: str) -> str:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    path = tempfile.mktemp(suffix=".docx")
    doc.save(path)
    return path

def generate_pdf(text: str) -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    path = tempfile.mktemp(suffix=".pdf")
    pdf.output(path)
    return path

# ---------------- PIPELINE ----------------

if process_btn and uploaded_file:
    with st.spinner("🔍 Performing AI OCR..."):
        image = Image.open(uploaded_file)
        extracted_text = jina_vision_ocr(image)

        if output_format == "Word (.docx)":
            output_path = generate_docx(extracted_text)
        else:
            output_path = generate_pdf(extracted_text)

        st.success("✅ OCR Completed Successfully")

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇ Download Output",
                f,
                file_name=os.path.basename(output_path)
            )

st.caption("Built with Cloud-Native AI & Human-Centered Design")
