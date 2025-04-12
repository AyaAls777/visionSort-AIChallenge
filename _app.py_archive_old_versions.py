"""
🗃️ ARCHIVED CODE — Not used in the final submitted app

This file contains earlier experimental versions and alternative implementations
of the VisionSort app. It includes:

- Initial UI structures that were later refactored
- GPT-4 prompt suggestion and fallback logic (commented out)
- BLIP captioning integration attempts (eventually removed)
- Other design variations and logic blocks

These sections were removed from main.py and app.py to simplify the final submission,
but are preserved here to document the development process, thought flow, and future plans.

Do not import or execute this file — it is for reference only.
"""


# #Imports
# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# import openai

# # Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # --- Streamlit Setup ---
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.sidebar.header("Configuration")

# # --- Sidebar Config ---
# min_confidence = st.sidebar.number_input("Confidence Threshold", min_value=0, max_value=100, value=25, step=1)
# borderline_min = st.sidebar.number_input("Borderline Minimum", min_value=0, max_value=100, value=15, step=1)


# # --- Main Interface ---
# st.title("🔍 VisionSort Pro")
# uploaded_files = st.file_uploader("Upload images/videos", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True)
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

# if uploaded_files and user_prompt:
#     results = {"high": [], "borderline": [], "low": []}
#     temp_paths = []

#     with st.spinner(f"Processing {len(uploaded_files)} files..."):
#         for file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#                 f.write(file.read())
#                 temp_paths.append(f.name)
#                 media_results = analyze_media(
#                     f.name, 
#                     user_prompt,
#                     min_confidence,
#                     (borderline_min, min_confidence)
#                 )

#                 for res in media_results:
#                     results[res["status"]].append(res)

#     # Sort all groups by confidence descending
#     for group in results.values():
#         group.sort(key=lambda r: r["confidence"], reverse=True)

#     # --- Display Confident Matches ---
#     if results["high"]:
#         st.subheader(f"🎯 Confident Matches ({len(results['high'])})")
#         cols = st.columns(4)
#         for idx, res in enumerate(results["high"]):
#             with cols[idx % 4]:
#                 st.image(Image.open(res["path"]), use_container_width=True)
#                 st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")

#     # --- Display Borderline Matches ---
#     if results["borderline"]:
#         st.subheader(f"⚠️ Potential Matches ({len(results['borderline'])})")
#         if st.checkbox("Show borderline results", True):
#             cols = st.columns(4)
#             for idx, res in enumerate(results["borderline"]):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")
#                     if st.button("🧠 Explain Match", key=f"blip_{idx}"):
#                         with st.expander("🔍 BLIP Analysis"):
#                             st.write(f"**BLIP Description:** {process_with_blip(res['path'])}")
#                             if "gpt_suggestion" in res:
#                                 st.write(f"**GPT Suggestion:** {res['gpt_suggestion']}")

#     # --- Display Low Confidence Matches Only If GPT Enabled ---
#     if results["low"] and openai.api_key:
#         st.subheader(f"❓ Low Confidence Matches ({len(results['low'])})")
#         if st.checkbox("Show low confidence results"):
#             for res in results["low"]:
#                 st.image(Image.open(res["path"]), use_container_width=True)
#                 st.caption(f"{res['confidence']:.1f}% | {res['timestamp']:.2f}s")
#                 if "gpt_suggestion" in res:
#                     st.markdown(f"**💡 GPT Suggestion:** {res['gpt_suggestion']}")

#     # --- Cleanup Temporary Files ---
#     for path in temp_paths:
#         if os.path.exists(path):
#             os.unlink(path)

#------original visonSort Chat-------------------------------------------------------
# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# import openai

# # Load OpenAI key from .env file
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # Set Streamlit layout
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.sidebar.header("Configuration")

# # === USER CONFIG ===
# min_confidence = st.sidebar.number_input(
#     "Confidence Threshold", min_value=0, max_value=100, value=25, step=1
# )

# # Helpful explanation
# st.sidebar.caption("💡 All results below the threshold will use fallback logic (BLIP/GPT).")

# # === UI: Upload Files ===
# st.title("🔍 VisionSort Pro")
# uploaded_files = st.file_uploader(
#     "Upload images or a video", 
#     type=["jpg", "jpeg", "png", "mp4", "mov"], 
#     accept_multiple_files=True,
#     key="file_uploader"
# )

# # Clear All Button
# if st.button("❌ Clear All"):
#     st.session_state["file_uploader"] = []  # Reset uploaded files

# # Prompt
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

# # === MEDIA TYPE CHECK ===
# if uploaded_files:
#     exts = {os.path.splitext(f.name)[1].lower() for f in uploaded_files}
#     if {".mp4", ".mov"}.intersection(exts) and {".jpg", ".jpeg", ".png"}.intersection(exts):
#         st.error("⚠️ Please upload only images OR only a video. Mixing is not supported.")
#         st.stop()

# # === MAIN LOGIC ===
# if uploaded_files and user_prompt:
#     temp_paths, results = [], {"confident": [], "fallback": []}

#     with st.spinner("Analyzing..."):
#         for file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#                 f.write(file.read())
#                 temp_paths.append(f.name)
#                 res = analyze_media(f.name, user_prompt, min_confidence)
#                 for r in res:
#                     group = "confident" if r["confidence"] >= min_confidence else "fallback"
#                     results[group].append(r)

#     # Sort all result groups high → low confidence
#     results["confident"].sort(key=lambda x: x["confidence"], reverse=True)
#     results["fallback"].sort(key=lambda x: x["confidence"], reverse=True)

#     # === DISPLAY: CONFIDENT MATCHES ===
#     if results["confident"]:
#         with st.expander(f"🎯 Confident Matches ({len(results['confident'])})", expanded=True):
#             cols = st.columns(4)
#             for idx, res in enumerate(results["confident"]):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")
#                     with st.expander("📌 Details"):
#                         st.write(f"**File:** {os.path.basename(res['path'])}")
#                         st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                         st.write(f"**Timestamp:** {res['timestamp']:.2f}s")
#                         st.write(f"**Location:** (Unavailable)")

#     # === DISPLAY: FALLBACK MATCHES ===
#     if results["fallback"]:
#         with st.expander(f"⚠️ Fallback Matches ({len(results['fallback'])})", expanded=True):
#             cols = st.columns(4)
#             for idx, res in enumerate(results["fallback"]):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")
#                     with st.expander("📌 Details"):
#                         st.write(f"**File:** {os.path.basename(res['path'])}")
#                         st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                         st.write(f"**Timestamp:** {res['timestamp']:.2f}s")
#                         st.write(f"**Location:** (Unavailable)")

#     # === FALLBACK: GPT PROMPT SUGGESTION ===
#     if not results["confident"] and openai.api_key:
#         st.markdown("---")
#         st.warning("😕 No confident results found.")
#         try:
#             captions = [process_with_blip(r["path"]) for r in results["fallback"][:3]]
#             suggestion_prompt = openai.ChatCompletion.create(
#                 model="gpt-4",
#                 messages=[
#                     {"role": "system", "content": "Suggest a clearer image prompt from captions."},
#                     {"role": "user", "content": "Captions:\n" + "\n".join(captions)}
#                 ],
#                 max_tokens=50
#             )
#             suggested = suggestion_prompt.choices[0].message.content.strip()
#             st.info(f"💡 Try this instead: **{suggested}**")
#         except Exception as e:
#             st.error(f"Error getting prompt suggestion: {str(e)}")

#     # === CLEANUP TEMP FILES ===
#     for path in temp_paths:
#         if os.path.exists(path):
#             os.remove(path)


#------updates visonsort chat--------------------------------------------------------------

# # ✅ Confidence confirmation button
# # ✅ Prompt auto-fill with GPT suggestion
# # ✅ Clear button appears only when media is uploaded
# # ✅ Sidebar UI cleaned up
# # ✅ App name centered
# # ✅ Loading spinner during Streamlit runs
# # ✅ Fallback cleanup logic (including crash safety)

# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# from dotenv import load_dotenv
# import openai

# # Load OpenAI key from .env file
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # --- Streamlit Setup ---
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.markdown("<h1 style='text-align: center;'>VisionSort Pro</h1>", unsafe_allow_html=True)

# # --- Sidebar UI ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.caption("Adjust filtering behavior before analyzing media.")
#     min_conf_slider = st.slider("Confidence Threshold", 0, 100, 25, step=1, key="threshold_slider")
#     confirm_threshold = st.button("Apply Threshold")
#     st.caption("Only frames above this confidence are shown as strong matches.")

# # Only update threshold when user confirms
# if confirm_threshold:
#     st.session_state["confirmed_threshold"] = st.session_state["threshold_slider"]
# elif "confirmed_threshold" not in st.session_state:
#     st.session_state["confirmed_threshold"] = 25
# min_confidence = st.session_state["confirmed_threshold"]

# # --- Upload Section ---
# st.markdown("---")
# uploaded_files = st.file_uploader(
#     "Upload images or a video", 
#     type=["jpg", "jpeg", "png", "mp4", "mov"], 
#     accept_multiple_files=True,
#     key="file_uploader"
# )

# if uploaded_files:
#     if st.button("❌ Clear All"):
#         st.session_state["file_uploader"] = []

# # Prompt
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'", key="user_prompt")

# # Media type check
# if uploaded_files:
#     exts = {os.path.splitext(f.name)[1].lower() for f in uploaded_files}
#     if {".mp4", ".mov"}.intersection(exts) and {".jpg", ".jpeg", ".png"}.intersection(exts):
#         st.error("⚠️ Please upload only images OR only a video. Mixing is not supported.")
#         st.stop()

# # Main app logic
# if uploaded_files and user_prompt:
#     temp_paths, results = [], {"confident": [], "fallback": []}

#     with st.spinner("Analyzing media..."):
#         for file in uploaded_files:
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#                     f.write(file.read())
#                     temp_paths.append(f.name)
#                     res = analyze_media(f.name, user_prompt, min_confidence)
#                     for r in res:
#                         group = "confident" if r["confidence"] >= min_confidence else "fallback"
#                         results[group].append(r)
#             except Exception as e:
#                 st.error(f"Failed to process file {file.name}: {e}")

#     results["confident"].sort(key=lambda x: x["confidence"], reverse=True)
#     results["fallback"].sort(key=lambda x: x["confidence"], reverse=True)

#     # Display confident matches
#     if results["confident"]:
#         with st.expander(f"🎯 Confident Matches ({len(results['confident'])})", expanded=True):
#             cols = st.columns(4)
#             for idx, res in enumerate(results["confident"]):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")
#                     with st.expander("📌 Details"):
#                         st.write(f"**File:** {os.path.basename(res['path'])}")
#                         st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                         st.write(f"**Timestamp:** {res['timestamp']:.2f}s")

#     # Display fallback matches
#     if results["fallback"]:
#         with st.expander(f"⚠️ Fallback Matches ({len(results['fallback'])})", expanded=True):
#             cols = st.columns(4)
#             for idx, res in enumerate(results["fallback"]):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")
#                     with st.expander("📌 Details"):
#                         st.write(f"**File:** {os.path.basename(res['path'])}")
#                         st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                         st.write(f"**Timestamp:** {res['timestamp']:.2f}s")

#     # Prompt suggestion fallback
#     if not results["confident"]:
#         st.warning("😕 No confident matches found.")
#         try:
#             captions = [process_with_blip(r["path"]) for r in results["fallback"][:3]]
#             suggestion_prompt = openai.ChatCompletion.create(
#              model="gpt-4",
#              messages=[
#              {"role": "system", "content": "Suggest a clearer image prompt from captions."},
#                 {"role": "user", "content": "Captions:\n" + "\n".join(captions)}
#                      ],
#                      max_tokens=50
# )

#             suggested = suggestion_prompt.choices[0].message.content.strip()
#             st.info(f"💡 Try this instead: **{suggested}**")
#             if st.button("Use Suggested Prompt"):
#                 st.session_state["user_prompt"] = suggested
#                 st.rerun()
#         except Exception as e:
#             st.error(f"Error generating prompt suggestion: {str(e)}")

#     # Clean up temp files even on crash
#     for path in temp_paths:
#         try:
#             if os.path.exists(path):
#                 os.remove(path)
#         except Exception as e:
#             st.warning(f"Couldn't delete temp file: {e}")

#-----------updated--------------------------------------------------
#analyze_media now takes frame_interval as a parameter (make sure main.py supports that)
# The Clear All fix avoids touching Streamlit widgets directly (you can’t modify file uploader state post-init)
# GPT fallback is untouched for now — we can re-add it in the fallback expander if you want
# Crash-safe cleanup: added try/except around os.remove() in case files are locked or used elsewhere
# Streamlit loading speed is mostly I/O-bound; slowing down is likely due to frame extraction or model loading. This optimization helps by skipping unnecessary reloading unless a button is clicked.

# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# from dotenv import load_dotenv
# import openai

# # Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # === App Setup ===
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.markdown("<h1 style='text-align: center;'>VisionSort Pro</h1>", unsafe_allow_html=True)

# # === Sidebar ===
# st.sidebar.title("⚙️ Configuration")

# # Confidence slider + apply button
# confidence = st.sidebar.slider("Confidence Threshold", 0, 100, 25)
# apply_conf = st.sidebar.button("Apply Threshold")

# # Frame sampling
# frame_interval = st.sidebar.slider("Video Frame Interval (1 = every frame)", 1, 120, 60)
# apply_frame = st.sidebar.button("Apply Frame Interval")

# # Store settings in session_state
# if apply_conf:
#     st.session_state["min_conf"] = confidence

# if apply_frame:
#     st.session_state["frame_interval"] = frame_interval

# # Set defaults if not set
# if "min_conf" not in st.session_state:
#     st.session_state["min_conf"] = 25
# if "frame_interval" not in st.session_state:
#     st.session_state["frame_interval"] = 60

# # === Upload Media ===
# uploaded_files = st.file_uploader(
#     "Upload images or a video",
#     type=["jpg", "jpeg", "png", "mp4", "mov"],
#     accept_multiple_files=True,
# )

# # Clear All
# if uploaded_files and st.button("❌ Clear All"):
#     uploaded_files.clear()  # Clear uploads
#     st.experimental_rerun()

# # Prompt
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the dog'")

# # === Main Logic ===
# if uploaded_files and user_prompt:
#     temp_paths, results = [], {"confident": [], "fallback": []}
#     st.info("⏳ Processing media... please wait.")
#     with st.spinner("Analyzing..."):
#         for file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#                 f.write(file.read())
#                 temp_paths.append(f.name)
#                 res = analyze_media(f.name, user_prompt,
#                                     min_confidence=st.session_state["min_conf"],
#                                     frame_interval=st.session_state["frame_interval"])
#                 for r in res:
#                     group = "confident" if r["confidence"] >= st.session_state["min_conf"] else "fallback"
#                     results[group].append(r)

#     results["confident"].sort(key=lambda x: x["confidence"], reverse=True)
#     results["fallback"].sort(key=lambda x: x["confidence"], reverse=True)

#     # === Confident Results ===
#     if results["confident"]:
#         st.subheader(f"🎯 Confident Matches ({len(results['confident'])})")
#         cols = st.columns(4)
#         for idx, res in enumerate(results["confident"]):
#             with cols[idx % 4]:
#                 st.image(Image.open(res["path"]), use_container_width=True)
#                 st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")

#     # === Fallback (Optional Reveal) ===
#     if results["fallback"]:
#         with st.expander(f"⚠️ Show Potential Matches ({len(results['fallback'])})"):
#             fallback_slider = st.slider("Show matches above this confidence", 0, st.session_state["min_conf"], 10)
#             filtered_fallback = [r for r in results["fallback"] if r["confidence"] >= fallback_slider]
#             filtered_fallback.sort(key=lambda x: x["confidence"], reverse=True)

#             cols = st.columns(4)
#             for idx, res in enumerate(filtered_fallback):
#                 with cols[idx % 4]:
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")

# # CLEANUP
# for path in temp_paths:
#     if os.path.exists(path):
#         try:
#             os.remove(path)
#         except Exception:
#             pass

#-------------------------------------------------------------------------------------------
# Metadata display on image click
# Smart frame interval UI (only for videos)
# Proper “Clear All” logic
# Removal of sidebar clutter
# A working “Apply Frame Interval” flow
# Confidence-based filtering with “Potential Matches” toggle
# Download option for selected images

# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# from dotenv import load_dotenv
# import openai
# import zipfile

# # Load OpenAI key from .env
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # App title
# st.set_page_config(layout="wide", page_title="VisionSort Pro")

# # Centered title
# st.markdown("<h1 style='text-align: center;'>VisionSort Pro</h1>", unsafe_allow_html=True)

# # === USER INPUT SECTION ===
# uploaded_files = st.file_uploader(
#     "Upload images or a video", 
#     type=["jpg", "jpeg", "png", "mp4", "mov"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     media_type = "video" if any(f.name.lower().endswith(('.mp4', '.mov')) for f in uploaded_files) else "image"
# else:
#     media_type = None

# # Only show frame interval if video uploaded
# if media_type == "video":
#     frame_interval = st.slider("Video Frame Interval (1 = every frame)", 1, 120, 30)
#     if st.button("Apply Frame Interval"):
#         st.session_state["frame_ready"] = True
# else:
#     frame_interval = None

# # Prompt
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

# # Clear All Button
# if st.button("❌ Clear All"):
#     st.session_state["clear_all"] = True

# if st.session_state.get("clear_all"):
#     uploaded_files = []
#     st.session_state["clear_all"] = False

# # === MAIN LOGIC ===
# if uploaded_files and user_prompt and (media_type == "image" or st.session_state.get("frame_ready")):
#     st.info("⏳ Processing media... please wait.")
#     temp_paths, all_results = [], []

#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#             f.write(file.read())
#             temp_path = f.name
#             temp_paths.append(temp_path)
#             res = analyze_media(temp_path, user_prompt, frame_interval=frame_interval if frame_interval else 30)
#             all_results.extend(res)

#     # Split results
#     min_confidence = 25
#     confident_results = [r for r in all_results if r["confidence"] >= min_confidence]
#     potential_results = [r for r in all_results if r["confidence"] < min_confidence]

#     # Hide processing state
#     st.empty()

#     # === CONFIDENT RESULTS ===
#     if confident_results:
#         st.subheader(f"🎯 Confident Matches ({len(confident_results)})")
#         selected = st.multiselect("Select images to download", [r["path"] for r in confident_results], key="confident_select")
#         if st.button("📥 Download Selected"):
#             zip_path = "selected_images.zip"
#             with zipfile.ZipFile(zip_path, "w") as zipf:
#                 for p in selected:
#                     zipf.write(p, arcname=os.path.basename(p))
#             with open(zip_path, "rb") as f:
#                 st.download_button("Download ZIP", f, file_name="selected_images.zip")
#             os.remove(zip_path)

#         cols = st.columns(4)
#         for idx, res in enumerate(confident_results):
#             with cols[idx % 4]:
#                 if st.button("Show Details", key=f"detail-{idx}"):
#                     st.image(Image.open(res["path"]), use_container_width=True)
#                     st.caption(f"🕒 {res['timestamp']:.2f}s | 📊 {res['confidence']:.1f}%")
#                     st.markdown(f"**File:** {os.path.basename(res['path'])}")
#                     st.markdown(f"**Confidence:** {res['confidence']:.1f}%")
#                     st.markdown(f"**Timestamp:** {res['timestamp']:.2f}s")

#     # === POTENTIAL RESULTS ===
#     if potential_results:
#         if st.checkbox("Show Potential Matches (below threshold)"):
#             min_potential = st.slider("Minimum confidence to show", 5, min_confidence - 1, 10)
#             filtered = [r for r in potential_results if r["confidence"] >= min_potential]
#             filtered.sort(key=lambda x: x["confidence"], reverse=True)

#             with st.expander(f"🌀 Potential Matches ({len(filtered)})", expanded=True):
#                 for r in filtered:
#                     try:
#                         caption = process_with_blip(r["path"])
#                         st.image(Image.open(r["path"]), use_container_width=True)
#                         st.caption(f"{caption}")
#                         st.write(f"🕒 {r['timestamp']:.2f}s | 📊 {r['confidence']:.1f}%")
#                     except Exception:
#                            st.write("⚠️ BLIP captioning failed")


#     # === CLEANUP TEMP FILES ===
#     for path in temp_paths:
#         if os.path.exists(path):
#             try:
#                 os.remove(path)
#             except Exception as e:
#                 st.warning(f"Could not delete: {path}")
#DEEPSEEK UPDATES-----------------------------------------------------------------------------------------------------
# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# from dotenv import load_dotenv
# import openai
# import zipfile
# import time

# # Load OpenAI key from .env
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # App title and config
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.markdown("<h1 style='text-align: center;'>VisionSort Pro</h1>", unsafe_allow_html=True)

# # Initialize session state for selections
# if 'selected_images' not in st.session_state:
#     st.session_state.selected_images = set()
# if 'selection_mode' not in st.session_state:
#     st.session_state.selection_mode = False

# # === USER INPUT SECTION ===
# uploaded_files = st.file_uploader(
#     "Upload images or a video", 
#     type=["jpg", "jpeg", "png", "mp4", "mov"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     media_type = "video" if any(f.name.lower().endswith(('.mp4', '.mov')) for f in uploaded_files) else "image"
# else:
#     media_type = None

# # Frame interval for videos
# frame_interval = st.slider("Video Frame Interval (frames to skip)", 1, 120, 30) if media_type == "video" else None

# # Prompt input
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

# # === MAIN PROCESSING ===
# if uploaded_files and user_prompt:
#     st.info("⏳ Processing media... please wait.")
#     temp_paths, all_results = [], []
    
#     progress_bar = st.progress(0)
#     for i, file in enumerate(uploaded_files):
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#             f.write(file.read())
#             temp_path = f.name
#             temp_paths.append(temp_path)
#             res = analyze_media(temp_path, user_prompt, frame_interval=frame_interval if frame_interval else 30)
#             all_results.extend(res)
#         progress_bar.progress((i + 1) / len(uploaded_files))
    
#     progress_bar.empty()
    
#     # Split results
#     min_confidence = 25
#     confident_results = [r for r in all_results if r["confidence"] >= min_confidence]
#     potential_results = [r for r in all_results if r["confidence"] < min_confidence]
    
#     # === SELECTION CONTROLS ===
#     col1, col2, col3 = st.columns([1, 1, 2])
#     with col1:
#         if st.button("🔘 Toggle Selection Mode"):
#             st.session_state.selection_mode = not st.session_state.selection_mode
#     with col2:
#         if st.session_state.selection_mode:
#             if st.button("📌 Select All"):
#                 st.session_state.selected_images.update(r["path"] for r in all_results)
#             if st.button("❌ Deselect All"):
#                 st.session_state.selected_images.clear()
    
#     # === CONFIDENT RESULTS ===
#     if confident_results:
#         st.subheader(f"🎯 Confident Matches ({len(confident_results)})")
        
#         # Display in 5-column grid
#         cols = st.columns(5)
#         for idx, res in enumerate(confident_results):
#             with cols[idx % 5]:
#                 img = Image.open(res["path"])
                
#                 # Selection overlay
#                 is_selected = res["path"] in st.session_state.selected_images
#                 if st.session_state.selection_mode:
#                     st.checkbox(
#                         f"Select {os.path.basename(res['path'])}", 
#                         value=is_selected,
#                         key=f"select_conf_{idx}",
#                         on_change=lambda idx=idx, path=res["path"]: st.session_state.selected_images.add(path) if st.session_state[f"select_conf_{idx}"] else st.session_state.selected_images.discard(path)
#                     )
                
#                 # Display image with optional selection highlight
#                 if is_selected:
#                     st.markdown("<div style='border: 3px solid #4CAF50; padding: 5px; border-radius: 5px;'>", unsafe_allow_html=True)
                
#                 st.image(img, use_container_width=True)
                
#                 if is_selected:
#                     st.markdown("</div>", unsafe_allow_html=True)
                
#                 # Show details button
#                 if st.button(f"Details {idx+1}", key=f"detail_conf_{idx}"):
#                     st.image(img, width=400)
#                     st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                     if res['timestamp'] > 0:
#                         mins, secs = divmod(res['timestamp'], 60)
#                         st.write(f"**Timestamp:** {int(mins):02d}:{secs:05.2f}")
#                     if res['datetime']:
#                         st.write(f"**Date Taken:** {res['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
    
#     # === POTENTIAL RESULTS ===
#     if potential_results:
#         with st.expander(f"🌀 Potential Matches ({len(potential_results)})", expanded=False):
#             # Display in 5-column grid
#             cols = st.columns(5)
#             for idx, res in enumerate(potential_results):
#                 with cols[idx % 5]:
#                     try:
#                         img = Image.open(res["path"])
                        
#                         # Selection overlay
#                         is_selected = res["path"] in st.session_state.selected_images
#                         if st.session_state.selection_mode:
#                             st.checkbox(
#                                 f"Select {os.path.basename(res['path'])}", 
#                                 value=is_selected,
#                                 key=f"select_pot_{idx}",
#                                 on_change=lambda idx=idx, path=res["path"]: st.session_state.selected_images.add(path) if st.session_state[f"select_pot_{idx}"] else st.session_state.selected_images.discard(path)
#                             )
                        
#                         # Display image with optional selection highlight
#                         if is_selected:
#                             st.markdown("<div style='border: 3px solid #FFA500; padding: 5px; border-radius: 5px;'>", unsafe_allow_html=True)
                        
#                         st.image(img, use_container_width=True)
                        
#                         if is_selected:
#                             st.markdown("</div>", unsafe_allow_html=True)
                        
#                         # Show details button
#                         if st.button(f"Details P{idx+1}", key=f"detail_pot_{idx}"):
#                             st.image(img, width=400)
#                             caption = process_with_blip(res["path"])
#                             st.write(f"**BLIP Caption:** {caption}")
#                             st.write(f"**Confidence:** {res['confidence']:.1f}%")
#                             if res['timestamp'] > 0:
#                                 mins, secs = divmod(res['timestamp'], 60)
#                                 st.write(f"**Timestamp:** {int(mins):02d}:{secs:05.2f}")
#                     except Exception as e:
#                         st.error(f"Error displaying image: {e}")
    
#     # === DOWNLOAD SELECTED ===
#     if st.session_state.selected_images:
#         if st.button("📥 Download Selected"):
#             zip_path = "selected_images.zip"
#             with zipfile.ZipFile(zip_path, "w") as zipf:
#                 for path in st.session_state.selected_images:
#                     if os.path.exists(path):
#                         zipf.write(path, arcname=os.path.basename(path))
            
#             with open(zip_path, "rb") as f:
#                 st.download_button(
#                     "Download ZIP", 
#                     f, 
#                     file_name="selected_images.zip",
#                     mime="application/zip"
#                 )
#             os.remove(zip_path)
    
#     # === CLEAR ALL BUTTON ===
#     if st.button("🧹 Clear All"):
#         st.session_state.clear()
#         st.experimental_rerun()
    
#     # Cleanup temp files
#     for path in temp_paths:
#         if os.path.exists(path):
#             try:
#                 os.remove(path)
#             except Exception as e:
#                 print(f"Could not delete temp file: {e}")

#GPT UPDATE-----------------------------------------------------------------------------------------------------------------------

# import os
# import tempfile
# import streamlit as st
# from main import analyze_media, process_with_blip
# from PIL import Image
# from datetime import datetime
# from dotenv import load_dotenv
# import shutil

# load_dotenv()
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # App layout config
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.markdown("<h1 style='text-align: center;'>VisionSort Pro</h1>", unsafe_allow_html=True)

# # Session state setup
# if "selection_mode" not in st.session_state:
#     st.session_state.selection_mode = False
# if "selected" not in st.session_state:
#     st.session_state.selected = set()
# if "clear_trigger" not in st.session_state:
#     st.session_state.clear_trigger = False

# # Sidebar was removed — all controls below the upload
# uploaded_files = st.file_uploader(
#     "Upload images or a video",
#     type=["jpg", "jpeg", "png", "mp4", "mov"],
#     accept_multiple_files=True,
#     key="media_upload"
# )

# # Frame Interval (only show if video is detected)
# frame_interval = 30
# if uploaded_files:
#     if any(file.name.endswith(('.mp4', '.mov')) for file in uploaded_files):
#         frame_interval = st.slider("Video Frame Interval (1 = every frame)", 1, 120, 30)
#         if st.button("Apply Frame Interval"):
#             st.session_state.frame_interval = frame_interval

# # Prompt
# user_prompt = st.text_input("Search prompt", placeholder="e.g. 'find the cat'")

# # Clear All button
# if st.button("❌ Clear All"):
#     st.session_state.clear_trigger = True
#     st.session_state.selected.clear()

# # Main analysis logic
# if st.session_state.clear_trigger:
#     uploaded_files = []
#     st.session_state.clear_trigger = False

# if uploaded_files and user_prompt:
#     st.info("⏳ Processing media... please wait.")
#     temp_paths = []
#     all_results = []

#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as f:
#             f.write(file.read())
#             temp_paths.append(f.name)
#             results = analyze_media(f.name, user_prompt, min_confidence=25, frame_interval=st.session_state.get("frame_interval", 30))
#             all_results.extend(results)

#     confident = [r for r in all_results if r["confidence"] >= 25]
#     potential = [r for r in all_results if r["confidence"] < 25]
#     confident.sort(key=lambda x: x["confidence"], reverse=True)
#     potential.sort(key=lambda x: x["confidence"], reverse=True)

#     # Global select toggle
#     st.subheader(f"🎯 Confident Matches ({len(confident)})")
#     col1, col2 = st.columns([1, 6])
#     with col1:
#         if st.button("Select"):
#             st.session_state.selection_mode = not st.session_state.selection_mode
#     with col2:
#         if st.session_state.selection_mode:
#             if st.button("Select All" if len(st.session_state.selected) < len(confident) + len(potential) else "Deselect All"):
#                 if len(st.session_state.selected) < len(confident) + len(potential):
#                     st.session_state.selected = {r["path"] for r in confident + potential}
#                 else:
#                     st.session_state.selected.clear()

#     # Download logic
#     if st.session_state.selection_mode and st.session_state.selected:
#         if st.download_button("⬇️ Download Selected", data=b"", file_name="selected_placeholder.txt"):
#             for path in st.session_state.selected:
#                 shutil.copy(path, os.path.join(os.getcwd(), os.path.basename(path)))

#     # Display confident matches
#     cols = st.columns(5)
#     for idx, r in enumerate(confident):
#         with cols[idx % 5]:
#             img = Image.open(r["path"])
#             if st.session_state.selection_mode:
#                 if st.button("✅" if r["path"] in st.session_state.selected else "☐", key=f"sel_{r['path']}"):
#                     if r["path"] in st.session_state.selected:
#                         st.session_state.selected.remove(r["path"])
#                     else:
#                         st.session_state.selected.add(r["path"])
#             st.image(img, use_container_width=True)
#             if st.button("Show Details", key=f"meta_conf_{idx}"):
#                 st.write(f"🕒 {r['timestamp']:.2f}s")
#                 st.write(f"📊 {r['confidence']:.1f}%")

#     # Low confidence section
#     if st.checkbox("Show Potential Matches (below threshold)"):
#         st.subheader(f"🌀 Potential Matches ({len(potential)})")
#         min_potential = st.slider("Minimum confidence to show", 1, 24, 10)
#         filtered = [r for r in potential if r["confidence"] >= min_potential]
#         cols = st.columns(5)
#         for idx, r in enumerate(filtered):
#             with cols[idx % 5]:
#                 try:
#                     caption = process_with_blip(r["path"])
#                     st.image(Image.open(r["path"]), use_container_width=True)
#                     st.caption(f"{caption}")
#                     st.caption(f"🕒 {r['timestamp']:.2f}s | 📊 {r['confidence']:.1f}%")
#                 except:
#                     st.caption("⚠️ BLIP captioning failed")

#     # Cleanup
#     for path in temp_paths:
#         if os.path.exists(path):
#             try:
#                 os.remove(path)
#             except Exception as e:
#                 st.warning(f"Could not delete: {path}")
#GPT cleanup new python---------------------------------------------------------------------------------------------------------------------------
# vision_sort_pro.py (COMPLETE: Spec-Matching Version)

# import os
# import tempfile
# import shutil
# from PIL import Image
# import streamlit as st
# from main import analyze_media, process_with_blip
# from sentence_transformers import SentenceTransformer, util
# from spellchecker import SpellChecker

# # Page Configuration
# st.set_page_config(layout="wide", page_title="VisionSort Pro")
# st.markdown("<h1 style='text-align: center;'>Vision Sort</h1>", unsafe_allow_html=True)

# # Init NLP models
# spell = SpellChecker()
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Session State Initialization
# if "selection_mode" not in st.session_state:
#     st.session_state.selection_mode = False
# if "selected" not in st.session_state:
#     st.session_state.selected = set()
# if "frame_interval" not in st.session_state:
#     st.session_state.frame_interval = 30

# # Upload & Media Handling
# uploaded_files = st.file_uploader("Upload images or a video", type=["jpg", "jpeg", "png", "mp4", "mov"], accept_multiple_files=True)
# mixed_upload = False
# video_uploaded = False

# if uploaded_files:
#     extensions = {os.path.splitext(f.name)[1].lower() for f in uploaded_files}
#     if any(ext in extensions for ext in [".mp4", ".mov"]):
#         video_uploaded = True
#     if len(extensions) > 1 and video_uploaded:
#         mixed_upload = True
#         st.error("🚨 Please upload either images *or* a video. Mixed uploads are not supported.")

#     if video_uploaded and not mixed_upload:
#         st.session_state.frame_interval = st.slider("Video Frame Interval (1 = every frame)", 1, 120, 30, key="video_interval")

# # Prompt Input Section
# user_prompt = st.text_input("Search for a scene or object...", placeholder="e.g. find the cat")


# # Clear Button (only shown if uploads exist)
# if uploaded_files:
#     if st.button("Clear All"):
#         st.session_state.selected.clear()
#         uploaded_files.clear()

# # Main Logic
# if uploaded_files and user_prompt and not mixed_upload:
#     st.info("⏳ Processing media... please wait.")
#     all_results, temp_paths = [], []

#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
#             tmp.write(file.read())
#             temp_paths.append(tmp.name)
#             results = analyze_media(tmp.name, user_prompt, frame_interval=st.session_state.frame_interval)
#             all_results.extend(results)

#     confident = [r for r in all_results if r["confidence"] >= 25]
#     potential = [r for r in all_results if 15 <= r["confidence"] < 25]
#     confident.sort(key=lambda x: x["confidence"], reverse=True)
#     potential.sort(key=lambda x: x["confidence"], reverse=True)

#     if not confident:
#         st.warning("No confident matches found. Want a closer look?")
#         st.session_state.show_potential = True

#     st.subheader(f"✅ Confident Matches ({len(confident)})")
#     if confident:
#         col1, col2 = st.columns([1, 6])
#         with col1:
#             st.session_state.selection_mode = st.toggle("Select Mode", value=st.session_state.selection_mode)
#         with col2:
#             if st.session_state.selection_mode:
#                 if st.button("Select All"):
#                     st.session_state.selected = {r["path"] for r in confident + potential}
#                 if st.button("Deselect All"):
#                     st.session_state.selected.clear()

#         cols = st.columns(5)
#         for idx, r in enumerate(confident):
#             with cols[idx % 5]:
#                 st.image(Image.open(r["path"]), use_container_width=True)
#                 if st.session_state.selection_mode:
#                     toggle_label = "✅" if r["path"] in st.session_state.selected else "☐"
#                     if st.button(toggle_label, key=f"select_{r['path']}"):
#                         if r["path"] in st.session_state.selected:
#                             st.session_state.selected.remove(r["path"])
#                         else:
#                             st.session_state.selected.add(r["path"])
#                 if st.button("Show Details", key=f"conf_details_{idx}"):
#                     st.write(f"🕒 {r['timestamp']:.2f}s | 📊 {r['confidence']:.1f}%")

#     # Show Low Confidence Section
#     show_potential = st.session_state.get("show_potential", False)
#     if show_potential or st.checkbox("⚠️ Show Potential Matches (below threshold)"):
#         min_thresh = st.slider("Min confidence to show", 15, 24, 20)
#         filtered = [r for r in potential if r["confidence"] >= min_thresh]
#         st.subheader(f"🌀 Potential Matches ({len(filtered)})")
#         cols = st.columns(5)
#         captions = []
#         for idx, r in enumerate(filtered):
#             with cols[idx % 5]:
#                 img = Image.open(r["path"])
#                 st.image(img, use_container_width=True)
#                 caption = process_with_blip(r["path"])
#                 captions.append((caption, r["path"]))
#                 st.caption(f"{caption}\n🕒 {r['timestamp']:.2f}s | 📊 {r['confidence']:.1f}%")

#----------#GPT RPROMPT TUNING-----------------------------------------------------------------------------------------------
#         # Prompt Tuning with GPT-like logic
#         corrected_prompt = " ".join([spell.correction(word) for word in user_prompt.split()])
#         user_embed = embedder.encode(corrected_prompt, convert_to_tensor=True)
#         caption_texts = [c[0] for c in captions]
#         caption_embeds = embedder.encode(caption_texts, convert_to_tensor=True)
#         sims = util.pytorch_cos_sim(user_embed, caption_embeds)[0]
#         ranked = sorted(zip(caption_texts, sims, captions), key=lambda x: x[1], reverse=True)
#         top_captions = [r[0] for r in ranked[:5]]

#         st.markdown("---")
#         st.markdown(f"**Prompt Assitant:**\nUser prompt: \"{user_prompt}\" → Corrected: \"{corrected_prompt}\"")
#         st.markdown("**Image Captions Most Similar:**")
#         for cap in top_captions:
#             st.markdown(f"- {cap}")

#         # Suggest new prompts
#         suggestions = [
#             f"Find a scene showing {cap.split()[0]}..." for cap in top_captions if len(cap.split()) > 1
#         ][:3]
#         if suggestions:
#             new_prompt = st.selectbox("💡 Try a refined prompt?", suggestions)
#             if st.button("🔁 Re-run with refined prompt"):
#                 st.experimental_rerun()

#     # Download Selected
#     if st.session_state.selection_mode and st.session_state.selected:
#         if st.download_button("⬇️ Download Selected", data=b"", file_name="selected_placeholder.txt"):
#             for path in st.session_state.selected:
#                 shutil.copy(path, os.path.join(os.getcwd(), os.path.basename(path)))

#     # Cleanup Temporary Files
#     for path in temp_paths:
#         if os.path.exists(path):
#             try:
#                 os.remove(path)
#             except Exception as e:
#                 st.warning(f"⚠️ Could not delete temporary file: {path}")

#------------GITHUB CODE ORIGINAL--------------------------------------------------------------------------------------------------------