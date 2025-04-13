#Imports
import os
import tempfile
import streamlit as st
from PIL import Image
# from main import analyze_media, process_with_blip
from main import analyze_media
#import openai
import io
import zipfile

# Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="Vision Sort")
st.sidebar.header("Configuration")

# --- Sidebar Config ---
min_confidence = st.sidebar.number_input("Confidence Threshold", min_value=0, max_value=100, value=25, step=1)
borderline_min = st.sidebar.number_input("Borderline Minimum", min_value=0, max_value=100, value=15, step=1)


# --- Main Interface ---
st.title("🔍 VisionSort Pro")
uploaded_files = st.file_uploader("Upload images/videos", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True)
user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

if uploaded_files and user_prompt:
    results = {"high": [], "borderline": [], "low": []}
    temp_paths = []

    with st.spinner(f"Processing {len(uploaded_files)} files..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
                f.write(file.read())
                temp_paths.append(f.name)
                media_results = analyze_media(
                    f.name, 
                    user_prompt,
                    min_confidence,
                    (borderline_min, min_confidence)
                )

                for res in media_results:
                    results[res["status"]].append(res)

    # Sort all groups by confidence descending
    for group in results.values():
        group.sort(key=lambda r: r["confidence"], reverse=True)

    # --- Display Confident Matches ---
    if results["high"]:
        st.subheader(f"🎯 Confident Matches ({len(results['high'])})")
        cols = st.columns(4)
        for idx, res in enumerate(results["high"]):
            with cols[idx % 4]:
                st.image(Image.open(res["path"]), use_container_width=True)
                st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")

    # --- Display Borderline Matches ---
    if results["borderline"]:
        st.subheader(f"⚠️ Potential Matches ({len(results['borderline'])})")
        #if st.checkbox("Show borderline results", True):
        if st.checkbox("Show borderline results", True, key="show_borderline"):
            cols = st.columns(4)
            for idx, res in enumerate(results["borderline"]):
                with cols[idx % 4]:
                    st.image(Image.open(res["path"]), use_container_width=True)
                    st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")
                    # if st.button("🧠 Explain Match", key=f"blip_{idx}"):
                        # with st.expander("🔍 BLIP Analysis"):
                        #     st.write(f"**BLIP Description:** {process_with_blip(res['path'])}")
                        #     if "gpt_suggestion" in res:
                        #         st.write(f"**GPT Suggestion:** {res['gpt_suggestion']}")

    # --- Display Low Confidence Matches Only If GPT Enabled ---
    # if results["low"] and openai.api_key:
    #     st.subheader(f"❓ Low Confidence Matches ({len(results['low'])})")
    #     if st.checkbox("Show low confidence results"):
    #         for res in results["low"]:
    #             st.image(Image.open(res["path"]), use_container_width=True)
    #             st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")
    #             if "gpt_suggestion" in res:
    #                 st.markdown(f"**💡 GPT Suggestion:** {res['gpt_suggestion']}")

    # --- Display Low Confidence Matches ------------------------------------------------------
    if results["low"]:
                st.subheader(f"❓ Low Confidence Matches ({len(results['low'])})")
               # if st.checkbox("Show low confidence results"):
                if st.checkbox("Show low confidence results", key="show_low"):
                    for res in results["low"]:
                        st.image(Image.open(res["path"]), use_container_width=True)
                        st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")

    # --- Prepare Downloadable Results ---
    download_ready = []

    if results["high"]:
        download_ready += results["high"]

    if results["borderline"] and st.session_state.get("show_borderline", True):
        download_ready += results["borderline"]

    if results["low"] and st.session_state.get("show_low", False):
        download_ready += results["low"]

    if download_ready:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for res in download_ready:
                try:
                    filename = os.path.basename(res["path"])
                    zipf.write(res["path"], arcname=filename)
                except Exception:
                    continue
        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Displayed Images",
            data=zip_buffer,
            file_name="visionSort_results.zip",
            mime="application/zip"
        )

        # --- Cleanup Temporary Files ---
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path) 
