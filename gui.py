import streamlit as st
import os
import requests
from duckduckgo_search import DDGS

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Image Scraper",
    page_icon="💝",
    layout="centered"
)

# =====================================================
# CUSTOM CSS (SEXY MODE ON)
# =====================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}
.title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #cfcfcf;
    margin-bottom: 30px;
}
label, .stSlider, .stTextInput {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# UI CARD
# =====================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title"><i>🖼 Image Scraper</i></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Turn keywords into ML-ready image datasets in one click</div>',
    unsafe_allow_html=True
)

# =====================================================
# INPUTS
# =====================================================
keyword = st.text_input("🔍 Dataset / Image Keyword", placeholder="e.g. wheat rust disease")

num_images = st.slider(
    "🖼️ Number of Images",
    min_value=10,
    max_value=500,
    value=100
)

ml_mode = st.toggle("🤖 Direct ML Dataset Creator Mode", value=True)

download_btn = st.button("🚀 Create Dataset")

# =====================================================
# DOWNLOAD FUNCTION
# =====================================================
def download_images(keyword, max_images, ml_mode, progress_bar, status):
    base_dir = "datasets\img-dataset" if ml_mode else "downloads"
    class_name = keyword.replace(" ", "_").lower()
    save_dir = os.path.join(base_dir, class_name)

    os.makedirs(save_dir, exist_ok=True)

    downloaded = 0

    with DDGS() as ddgs:
        results = ddgs.images(
            keywords=keyword,
            region="wt-wt",
            safesearch="off",
            max_results=max_images
        )

        for i, r in enumerate(results):
            try:
                img_url = r["image"]
                img_data = requests.get(img_url, timeout=10).content

                file_path = os.path.join(
                    save_dir, f"img_{downloaded+1:04d}.jpg"
                )

                with open(file_path, "wb") as f:
                    f.write(img_data)

                downloaded += 1

            except:
                pass

            progress_bar.progress((i + 1) / max_images)
            status.text(f"Downloading {i+1}/{max_images}")

    return downloaded, save_dir

# =====================================================
# BUTTON ACTION
# =====================================================
if download_btn:
    if keyword.strip() == "":
        st.error("❌ Keyword cannot be empty")
    else:
        st.success("⚡ Dataset creation started")
        progress = st.progress(0.0)
        status = st.empty()

        total, path = download_images(
            keyword, num_images, ml_mode, progress, status
        )

        progress.progress(1.0)
        status.text("✅ Completed")

        st.success(f"🎉 {total} images downloaded")
        st.info(f"📂 Dataset path: `{path}`")

        if ml_mode:
            st.markdown("### ✅ ML Ready")
            st.markdown(
                "- Compatible with **TensorFlow / PyTorch**  \n"
                "- Folder = class label  \n"
                "- No manual cleaning required"
            )

st.markdown('</div>', unsafe_allow_html=True)
