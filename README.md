# VisionSort  
*AI-powered visual search tool for finding key moments in large batches of images or video frames using natural prompts.*

---

## Concept Summary

VisionSort helps users avoid manually scrubbing through thousands of images or long video footage. Instead, they can simply describe what they’re looking for in natural language — like:

> “Show me images with meteors,”  
> “Find the person wearing a blue hoodie,”  
> “Only frames with the cat near the window.”

The app uses OpenAI's CLIP model to semantically compare your prompt to the visual content of uploaded images or video frames. It then displays the most relevant matches with ranked confidence scores and video timestamps.

---

## Target Users

- **Astrophotographers & skywatchers** — spotting rare meteor events
- **Surveillance teams / CCTV users** — locating key moments in footage
- **Researchers or satellite image analysts** — filtering massive visual datasets
- **Drone operators or hobbyists** — identifying key subjects
- **Anyone with a large photo/video archive** — looking for specific visuals

---

## Tech Stack

- VS Code (development)
- Jupyter Notebook (early prototyping)
- Python 3.11.11
- Streamlit (app UI)
- OpenAI CLIP (ViT-B/32 model for image-text matching)
- OpenCV (video frame extraction)
- Pillow (image handling and processing)


---


## Key Features

- Upload multiple images or videos  
- Auto-extract frames from video (1 frame/sec)  
- Search using natural language prompts  
- Semantic similarity matching using CLIP embeddings + cosine similarity  
- Results sorted into:  
  - 🎯 Confident Matches  
  - ⚠️ Potential Matches (borderline)  
  - ❓ Low Confidence Matches  
- Interactive Configuration Panel:  
  - Adjust confidence threshold and borderline minimum  
  - Toggle display of borderline and low-confidence results  
- Timestamp support for video frames  
- Download Displayed Results as `.zip` based on current filter settings  
- Temp file cleanup on each run


---


## Archived & Upcoming Features

This version focuses on a clean, working CLIP-based prototype. 

The following features were previously implemented but later removed  (archived in _main.py_archive_old_versions.py and _app.py_archive_old_versions.py)
  to improve performance and simplify the user experience — but are preserved for future updates:

- GPT-4 integration for prompt refinement when user input was vague or misspelled  
- User-controlled frame sampling rate (choose how many frames to extract from videos)  
- Optional fallback triggers — user could decide when to use BLIP or GPT help  
- Alternative UI versions with more interactive elements



---

## Challenges Faced

-Deployment issues on Streamlit Cloud due to Python versioning, OpenCV, and Torch compatibility, had to switch to hugging face for deployment.
- Balancing scope under tight time pressure.
- First time independently building an AI project — and seriously working with Python.
- Initially aimed to integrate CLIP (semantic search), GPT-4 (prompt refinement), and BLIP (fallback captioning) — but the stack proved too complex for the challenge timeline.
- Learned an important lesson in **scope control** under tight deadlines.
- Faced performance issues when processing large batches of images and videos, which taught me the need to write code that handles batch operations efficiently.
- Experimented with BLIP as a fallback model:
  - Helped add context, but often lacked precision.
  - Highlighted the need for smarter fallback triggers and sparked interest in future models like PaLI or GIT.
- Streamlit-specific challenges:
  - Managing multiple file uploads and temp files  
  - Keeping UI responsive with real-time feedback (match scores, timestamps, toggles)
- Key takeaway: **Build a stable core first**, then layer in advanced features.
- Although GPT-4 and BLIP weren’t fully integrated in the final version, I preserved and documented their experiments for future improvements.



