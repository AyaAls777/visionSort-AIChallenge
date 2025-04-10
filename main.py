#Imports
import os
import cv2
import torch
import clip
import openai
from PIL import Image
from datetime import datetime
from functools import lru_cache
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize OpenAI API
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Video processing
def extract_frames(video_path, frame_interval=30):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, total_frames, frame_interval):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = vidcap.read()
        if success:
            frame_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
    vidcap.release()
    return frames, fps

@lru_cache(maxsize=100)
def process_with_blip(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device)
        caption = blip_model.generate(**inputs, max_new_tokens=50)[0]
        return blip_processor.decode(caption, skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_media(file_path, prompt, min_confidence=25, borderline_range=(15,25)):
    # Handle both images and videos
    if file_path.endswith(('.mp4', '.mov')):
        frame_paths, fps = extract_frames(file_path)
        timestamps = [i/fps for i in range(0, len(frame_paths)*30, 30)]
    else:
        frame_paths = [file_path]
        timestamps = [0]
    
    results = []
    for path, timestamp in zip(frame_paths, timestamps):
        # CLIP analysis
        image = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
        
        confidence = similarity.item() * 100
        result = {
            "path": path,
            "confidence": confidence,
            "timestamp": timestamp,
            "source": "CLIP",
            "status": (
                "high_confidence" if confidence >= min_confidence else
                "borderline" if confidence >= borderline_range[0] else
                "low_confidence"
            )
        }
        
        # Only use GPT-4 for very low confidence if available
        if confidence < borderline_range[0] and openai.api_key:
            try:
                blip_desc = process_with_blip(path)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "Suggest one improved image search prompt based on:"
                    }, {
                        "role": "user",
                        "content": blip_desc
                    }],
                    max_tokens=50
                )
                result["gpt_suggestion"] = response.choices[0].message.content
            except:
                pass
        
        results.append(result)
    
    return results