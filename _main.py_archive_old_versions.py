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
# import cv2
# import torch
# import clip
# import openai
# from PIL import Image
# from datetime import datetime
# from functools import lru_cache
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # Initialize models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# # Video processing
# def extract_frames(video_path, frame_interval=30):
#     frames = []
#     vidcap = cv2.VideoCapture(video_path)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     for i in range(0, total_frames, frame_interval):
#         vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         success, frame = vidcap.read()
#         if success:
#             frame_path = f"temp_frame_{i}.jpg"
#             cv2.imwrite(frame_path, frame)
#             frames.append(frame_path)
#     vidcap.release()
#     return frames, fps

# @lru_cache(maxsize=100)
# def process_with_blip(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = blip_processor(image, return_tensors="pt").to(device)
#         caption = blip_model.generate(**inputs, max_new_tokens=50)[0]
#         return blip_processor.decode(caption, skip_special_tokens=True)
#     except Exception as e:
#         return f"Error: {str(e)}"
    
# def analyze_media(file_path, prompt, min_confidence=25):
#     # Handle both images and videos
#     if file_path.endswith(('.mp4', '.mov')):
#         frame_paths, fps = extract_frames(file_path)
#         timestamps = [i/fps for i in range(0, len(frame_paths)*30, 30)]
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     results = []
#     for path, timestamp in zip(frame_paths, timestamps):
#         try:
#             image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt]).to(device)

#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)
#                 text_features = clip_model.encode_text(text)
#                 similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

#             confidence = similarity.item() * 100
#             result = {
#                 "path": path,
#                 "confidence": confidence,
#                 "timestamp": timestamp,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback"
#             }
#             results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Processing frame failed: {e}")
#     return results


#------updates^ original visonSort chat--------------------------------------------------------------
# We can simplify analyze_media() like this:

# ✅ Key Changes:
# diff
# Copy
# Edit
# def analyze_media(file_path, prompt, min_confidence=25):
# -    borderline_range = (15, 25)  # ❌ remove this
#      ...
# -        "status": (
# -            "high_confidence" if confidence >= min_confidence else
# -            "borderline" if confidence >= borderline_range[0] else
# -            "low_confidence"
# -        )
# +        "status": "confident" if confidence >= min_confidence else "fallback"
#This will align it with the refactored logic in app.py, making your data flow more consistent and easier to debug.#

#-------------------Below updates visionsort chat-----------------------------------------------------------------------------
# analyze_media() now accepts and passes frame_interval directly to extract_frames()
# Frame timestamps are correctly calculated based on your chosen interval
# Still supports both images and videos without breaking compatibility
# Cleanup-safe and GPU-friendly if available
# import os
# import cv2
# import torch
# import clip
# import openai
# from PIL import Image
# from datetime import datetime
# from functools import lru_cache
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # Init device & models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# # === Video Frame Extractor ===
# def extract_frames(video_path, frame_interval=60):
#     frames = []
#     vidcap = cv2.VideoCapture(video_path)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

#     for i in range(0, total_frames, frame_interval):
#         vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         success, frame = vidcap.read()
#         if success:
#             frame_path = f"temp_frame_{i}.jpg"
#             cv2.imwrite(frame_path, frame)
#             frames.append(frame_path)
#     vidcap.release()
#     return frames, fps

# # === BLIP Captioning ===
# @lru_cache(maxsize=100)
# def process_with_blip(image_path):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = blip_processor(image, return_tensors="pt").to(device)
#         caption = blip_model.generate(**inputs, max_new_tokens=50)[0]
#         return blip_processor.decode(caption, skip_special_tokens=True)
#     except Exception as e:
#         return f"Error: {str(e)}"

# === Main Inference Logic ===
# def analyze_media(file_path, prompt, min_confidence=25, frame_interval=60):
#     # Choose logic based on media type
#     if file_path.endswith(('.mp4', '.mov')):
#         frame_paths, fps = extract_frames(file_path, frame_interval)
#         timestamps = [i/fps for i in range(0, len(frame_paths)*frame_interval, frame_interval)]
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     results = []
#     for path, timestamp in zip(frame_paths, timestamps):
#         try:
#             image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt]).to(device)

#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)
#                 text_features = clip_model.encode_text(text)
#                 similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

#             confidence = similarity.item() * 100
#             result = {
#                 "path": path,
#                 "confidence": confidence,
#                 "timestamp": timestamp,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback"
#             }
#             results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Failed on {path}: {e}")
#     return results


# def analyze_media(file_path, prompt, min_confidence=25, frame_interval=30):
#     # Handle both images and videos
#     if file_path.endswith(('.mp4', '.mov')):
#         frame_paths, fps = extract_frames(file_path, frame_interval)
#         timestamps = [i / fps for i in range(len(frame_paths))]
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     results = []
#     for path, timestamp in zip(frame_paths, timestamps):
#         try:
#             image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt]).to(device)

#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)
#                 text_features = clip_model.encode_text(text)
#                 similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

#             confidence = similarity.item() * 100
#             result = {
#                 "path": path,
#                 "confidence": confidence,
#                 "timestamp": timestamp,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback"
#             }
#             results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Processing frame failed: {e}")
#     return results

#DEEPSEEK UPDATES testing------------------------------------------------------------------------------------------------------------
# import os
# import cv2
# import torch
# import clip
# import openai
# from PIL import Image, ExifTags
# from datetime import datetime
# from functools import lru_cache
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Initialize OpenAI API
# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = api_key

# # Init device & models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)  # Fix for warning
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# def get_image_datetime(image_path):
#     """Extract datetime from image EXIF data if available"""
#     try:
#         img = Image.open(image_path)
#         if hasattr(img, '_getexif'):
#             exif = img._getexif()
#             if exif:
#                 for tag, value in exif.items():
#                     if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'DateTimeOriginal':
#                         return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
#     except Exception:
#         pass
#     return None

# def extract_frames(video_path, frame_interval=30):
#     """Improved video frame extraction with better error handling"""
#     frames = []
#     timestamps = []
    
#     try:
#         vidcap = cv2.VideoCapture(video_path)
#         if not vidcap.isOpened():
#             raise ValueError(f"Could not open video file: {video_path}")
            
#         fps = vidcap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         for i in range(0, total_frames, frame_interval):
#             vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             success, frame = vidcap.read()
#             if success:
#                 frame_path = f"temp_frame_{i}.jpg"
#                 cv2.imwrite(frame_path, frame)
#                 frames.append(frame_path)
#                 timestamps.append(i / fps)
                
#         vidcap.release()
#         return frames, timestamps
        
#     except Exception as e:
#         print(f"[ERROR] Video processing failed: {e}")
#         if 'vidcap' in locals():
#             vidcap.release()
#         return [], []

# @lru_cache(maxsize=100)
# def process_with_blip(image_path):
#     """BLIP captioning with better error handling"""
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = blip_processor(image, return_tensors="pt").to(device)
#         caption = blip_model.generate(**inputs, max_new_tokens=50)[0]
#         return blip_processor.decode(caption, skip_special_tokens=True)
#     except Exception as e:
#         print(f"[BLIP Error] {str(e)}")
#         return "Could not generate caption"

# def analyze_media(file_path, prompt, min_confidence=25, frame_interval=30):
#     """Improved media analysis with better metadata handling"""
#     # Handle both images and videos
#     if file_path.lower().endswith(('.mp4', '.mov')):
#         frame_paths, timestamps = extract_frames(file_path, frame_interval)
#         if not frame_paths:
#             return []
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     results = []
#     for path, timestamp in zip(frame_paths, timestamps):
#         try:
#             image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt]).to(device)

#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)
#                 text_features = clip_model.encode_text(text)
#                 similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

#             confidence = similarity.item() * 100
#             datetime_info = get_image_datetime(path) if not file_path.lower().endswith(('.mp4', '.mov')) else None
            
#             result = {
#                 "path": path,
#                 "confidence": confidence,
#                 "timestamp": timestamp,
#                 "datetime": datetime_info,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback"
#             }
#             results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Processing frame failed: {e}")
#     return results
#GPT bugs UPDATE---------------------------------------------------------------------------------------------------------------
# main.py
# import os
# import cv2
# import torch
# import clip
# import openai
# from PIL import Image
# from datetime import datetime
# from functools import lru_cache
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from dotenv import load_dotenv

# # Load .env for OpenAI
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Init models (lazy loaded for performance)
# device = "cuda" if torch.cuda.is_available() else "cpu"

# @lru_cache(maxsize=1)
# def get_clip_model():
#     return clip.load("ViT-B/32", device=device)

# @lru_cache(maxsize=1)
# def get_blip_models():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
#     return processor, model

# # Video frame extraction
# def extract_frames(video_path, frame_interval=30):
#     frames = []
#     vidcap = cv2.VideoCapture(video_path)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

#     for i in range(0, total_frames, frame_interval):
#         vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         success, frame = vidcap.read()
#         if success:
#             frame_path = f"temp_frame_{i}.jpg"
#             cv2.imwrite(frame_path, frame)
#             frames.append(frame_path)
#     vidcap.release()
#     return frames, fps

# # BLIP fallback
# @lru_cache(maxsize=100)
# def process_with_blip(image_path):
#     processor, model = get_blip_models()
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = processor(image, return_tensors="pt").to(device)
#         caption_ids = model.generate(**inputs, max_new_tokens=50)[0]
#         return processor.decode(caption_ids, skip_special_tokens=True)
#     except Exception as e:
#         return f"BLIP error: {str(e)}"

# # Core logic
# def analyze_media(file_path, prompt, min_confidence=25, frame_interval=30):
#     clip_model, clip_preprocess = get_clip_model()
    
#     # Handle video vs image
#     if file_path.endswith(('.mp4', '.mov', '.mpeg4')):
#         frame_paths, fps = extract_frames(file_path, frame_interval)
#         timestamps = [i / fps for i in range(len(frame_paths))]
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     results = []
#     for path, timestamp in zip(frame_paths, timestamps):
#         try:
#             image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
#             text = clip.tokenize([prompt]).to(device)
#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)
#                 text_features = clip_model.encode_text(text)
#                 similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
#             confidence = similarity.item() * 100
#             results.append({
#                 "path": path,
#                 "confidence": confidence,
#                 "timestamp": timestamp,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback"
#             })
#         except Exception as e:
#             print(f"[ERROR] Processing frame failed: {e}")
#     return results

#GPT cleanup new python---------------------------------------------------------------------------------------------------------------------------
# main.py (COMPLETE: Spec-Matching Version)
# main.py (Refactored for batching, async, EXIF)

# main.py (Refactored for batching, async, EXIF, and video fix)

# main.py (Optimized: Max 60 frames, 1 FPS, Removed Interval Slider)

# import os
# import cv2
# import torch
# import clip
# import openai
# import asyncio
# import concurrent.futures
# from PIL import Image, UnidentifiedImageError, ExifTags
# from datetime import datetime
# from functools import lru_cache
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from dotenv import load_dotenv
# from torchvision import transforms

# # Load API Keys
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Device Setup
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Init Models (Lazy Cache)
# @lru_cache(maxsize=1)
# def get_clip_model():
#     return clip.load("ViT-B/32", device=device)

# @lru_cache(maxsize=1)
# def get_blip_models():
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
#     return processor, model

# # Extract up to 60 frames at 1 FPS
# def extract_frames(video_path):
#     frames = []
#     vidcap = cv2.VideoCapture(video_path)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     interval = int(fps)  # 1 frame per second
#     max_frames = 60

#     for i in range(0, min(total_frames, max_frames * interval), interval):
#         vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         success, frame = vidcap.read()
#         if success:
#             frame_path = f"temp_frame_{i}.jpg"
#             cv2.imwrite(frame_path, frame)
#             frames.append(frame_path)
#     vidcap.release()
#     return frames, fps

# # BLIP fallback captioning
# @lru_cache(maxsize=100)
# def process_with_blip(image_path):
#     processor, model = get_blip_models()
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = processor(image, return_tensors="pt").to(device)
#         caption_ids = model.generate(**inputs, max_new_tokens=50)[0]
#         return processor.decode(caption_ids, skip_special_tokens=True)
#     except Exception as e:
#         return f"BLIP error: {str(e)}"

# # Optional EXIF extractor
# def extract_metadata(image_path):
#     try:
#         image = Image.open(image_path)
#         exif_data = image._getexif()
#         if not exif_data:
#             return {}
#         labeled = {
#             ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()
#             if k in ExifTags.TAGS
#         }
#         return labeled
#     except Exception:
#         return {}

# # Resize & preprocess
# clip_resize = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Batch processing helper
# def get_clip_features_batch(image_paths, model, preprocess, batch_size=32):
#     images = []
#     for p in image_paths:
#         try:
#             img = preprocess(Image.open(p).convert("RGB"))
#             images.append(img)
#         except UnidentifiedImageError:
#             continue  # Skip bad frames
#     if not images:
#         return torch.empty(0)
#     image_batches = [torch.stack(images[i:i+batch_size]) for i in range(0, len(images), batch_size)]
#     encoded = []
#     with torch.no_grad():
#         for batch in image_batches:
#             encoded.append(model.encode_image(batch.to(device)))
#     return torch.cat(encoded)

# # Async helper
# async def run_async_batches(func, items):
#     loop = asyncio.get_event_loop()
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         return await asyncio.gather(*[loop.run_in_executor(pool, func, *item) for item in items])

# # Main media analysis logic
# def analyze_media(file_path, prompt, min_confidence=25):
#     clip_model, clip_preprocess = get_clip_model()
#     frame_paths = []
#     timestamps = []

#     # Detect if video
#     if file_path.endswith((".mp4", ".mov", ".mpeg4")):
#         frame_paths, fps = extract_frames(file_path)
#         timestamps = [i for i in range(len(frame_paths))]  # 1 second per frame
#     else:
#         frame_paths = [file_path]
#         timestamps = [0]

#     # Prepare text features
#     text = clip.tokenize([prompt]).to(device)
#     with torch.no_grad():
#         text_features = clip_model.encode_text(text)

#     # Batch encode images
#     image_features = get_clip_features_batch(frame_paths, clip_model, clip_preprocess)
#     if image_features.shape[0] == 0:
#         return []

#     results = []
#     for idx, (img_path, img_feat, ts) in enumerate(zip(frame_paths, image_features, timestamps)):
#         sim = torch.nn.functional.cosine_similarity(img_feat.unsqueeze(0), text_features)
#         confidence = sim.item() * 100
#         if confidence >= 15:
#             results.append({
#                 "path": img_path,
#                 "confidence": confidence,
#                 "timestamp": ts,
#                 "source": "CLIP",
#                 "status": "confident" if confidence >= min_confidence else "fallback",
#                 "metadata": extract_metadata(img_path)
#             })
#     return results

#     return results
