# # app.py (clean, production-friendly MVP)
# import os
# import io
# import time
# import json
# import tempfile
# import requests
# import numpy as np
# import streamlit as st
# from pydub import AudioSegment, silence
# import cv2
# import shutil
# import subprocess
# import random
# import hashlib
# from collections import defaultdict, Counter

# # ---------------------------
# # CONFIG / SECRETS
# # ---------------------------
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
# HF_API_KEY = st.secrets.get("HF_API_KEY", None)

# if not OPENAI_API_KEY:
#     st.error("OpenAI API key missing in .streamlit/secrets.toml or environment.")
#     st.stop()

# OPENAI_BASE = "https://api.openai.com/v1"
# HEADERS_AUTH = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# # ---------------------------
# # UTIL: file helpers
# # ---------------------------
# def save_uploaded_file(uploaded_file, dst_path):
#     with open(dst_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return dst_path

# # ---------------------------
# # AUDIO / VIDEO UTIL (ffmpeg CLI)
# # ---------------------------
# # Попробуй найти системный ffmpeg, иначе используем imageio-ffmpeg
# def check_ffmpeg_available():
#     """Return path to ffmpeg if available. Try system ffmpeg first, then imageio-ffmpeg."""
#     # 1) system ffmpeg on PATH
#     sys_path = shutil.which("ffmpeg")
#     if sys_path:
#         return sys_path

#     # 2) try imageio-ffmpeg package binary
#     try:
#         from imageio_ffmpeg import get_ffmpeg_exe
#         exe = get_ffmpeg_exe()  # returns path to bundled ffmpeg binary
#         if exe:
#             return exe
#     except Exception:
#         pass

#     # not found
#     return None


# def extract_audio_from_video(video_path, out_audio_path, sample_rate=16000):
#     ffmpeg_path = check_ffmpeg_available()
#     if not ffmpeg_path:
#         raise RuntimeError(
#             "ffmpeg is not found in PATH. Please install system ffmpeg.\n"
#             "macOS: brew install ffmpeg\n"
#             "Ubuntu/Debian: sudo apt install ffmpeg\n"
#             "Windows: install ffmpeg and add to PATH"
#         )
#     cmd = [
#         ffmpeg_path,
#         "-y",
#         "-i", video_path,
#         "-vn",
#         "-acodec", "pcm_s16le",
#         "-ac", "1",
#         "-ar", str(sample_rate),
#         out_audio_path
#     ]
#     subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     if not (os.path.exists(out_audio_path) and os.path.getsize(out_audio_path) > 0):
#         raise RuntimeError("Audio extraction completed but output file is missing or empty.")
#     return out_audio_path

# def detect_silences(audio_path, min_silence_len_ms=1000, silence_thresh_db=-40):
#     audio = AudioSegment.from_wav(audio_path)
#     silences = silence.detect_silence(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
#     silences_sec = [(s/1000.0, e/1000.0) for s,e in silences]
#     return silences_sec

# # ---------------------------
# # Transcription via REST (Whisper)
# # ---------------------------
# def transcribe_with_whisper_rest(audio_path):
#     url = f"{OPENAI_BASE}/audio/transcriptions"
#     with open(audio_path, "rb") as f:
#         files = {"file": ("audio.wav", f, "audio/wav")}
#         data = {"model": "whisper-1"}
#         headers = HEADERS_AUTH.copy()
#         try:
#             resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
#             resp.raise_for_status()
#             j = resp.json()
#         except Exception as e:
#             st.error(f"Whisper transcription error: {e}")
#             return {"text": "", "segments": []}
#     text = j.get("text", "") if isinstance(j, dict) else ""
#     segments = j.get("segments", []) if isinstance(j, dict) else []
#     return {"text": text, "segments": segments}

# # ---------------------------
# # FRAME SAMPLING & FACE DETECTION
# # ---------------------------
# HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# def sample_frames(video_path, sample_rate_sec=2.0, max_frames=200):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("Could not open video file for frame sampling.")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     duration = frame_count / fps if fps else 0
#     frames = []
#     if duration == 0:
#         while len(frames) < max_frames:
#             success, frame = cap.read()
#             if not success:
#                 break
#             t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#             frames.append((float(t), frame.copy()))
#         cap.release()
#         return frames
#     times = np.arange(0, duration, sample_rate_sec)
#     for t in times:
#         cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
#         success, frame = cap.read()
#         if not success:
#             continue
#         frames.append((float(t), frame.copy()))
#         if len(frames) >= max_frames:
#             break
#     cap.release()
#     return frames

# def detect_faces_in_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(HAAR_XML)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
#     return faces

# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
#     yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     interArea = interW * interH
#     boxAArea = boxA[2]*boxA[3]
#     boxBArea = boxB[2]*boxB[3]
#     union = boxAArea + boxBArea - interArea
#     if union == 0:
#         return 0
#     return interArea / union

# def track_faces(frames_with_faces, iou_thresh=0.3):
#     tracks = {}
#     next_id = 1
#     timeline = []
#     for t, boxes in frames_with_faces:
#         row = []
#         for box in boxes:
#             assigned = False
#             for tid, last_box in tracks.items():
#                 if iou(box, last_box) > iou_thresh:
#                     tracks[tid] = box
#                     row.append((tid, box))
#                     assigned = True
#                     break
#             if not assigned:
#                 tracks[next_id] = box
#                 row.append((next_id, box))
#                 next_id += 1
#         timeline.append((t, row))
#     return timeline

# # ---------------------------
# # HUGGINGFACE EMOTION INFERENCE (cache + parser)
# # ---------------------------
# HF_EMOTION_MODEL = "dima806/facial_emotions_image_detection"
# _hf_cache = {}

# def hf_classify_image_raw_jpeg(model_name, image_bytes_io, hf_token, timeout=60):
#     api_url = f"https://api-inference.huggingface.co/models/{model_name}"
#     headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
#     headers["Content-Type"] = "image/jpeg"
#     try:
#         resp = requests.post(api_url, headers=headers, data=image_bytes_io.getvalue(), timeout=timeout)
#     except requests.exceptions.Timeout as e:
#         return {"ok": False, "status_code": None, "text": "", "json": None, "error": f"timeout: {e}"}
#     except Exception as e:
#         return {"ok": False, "status_code": None, "text": "", "json": None, "error": f"request error: {e}"}
#     text = resp.text[:4000]
#     try:
#         parsed = resp.json()
#     except Exception:
#         parsed = None
#     if resp.status_code != 200:
#         return {"ok": False, "status_code": resp.status_code, "text": text, "json": parsed, "error": None}
#     return {"ok": True, "status_code": resp.status_code, "text": text, "json": parsed, "error": None}

# def hf_infer_with_cache(model_name, img_bytes_io, hf_token):
#     key = hashlib.sha256(img_bytes_io.getvalue()).hexdigest()
#     cache_key = f"{model_name}:{key}"
#     if cache_key in _hf_cache:
#         return _hf_cache[cache_key]
#     dbg = hf_classify_image_raw_jpeg(model_name, img_bytes_io, hf_token)
#     _hf_cache[cache_key] = dbg
#     return dbg

# def classify_face_emotion(crop_bgr, hf_token, model_name):
#     _, buf = cv2.imencode('.jpg', crop_bgr)
#     img_bytes = io.BytesIO(buf.tobytes())
#     if not hf_token:
#         return {"label": "unknown", "score": 0.0, "raw": None, "source": "none"}
#     dbg = hf_infer_with_cache(model_name, img_bytes, hf_token)
#     if not dbg or not dbg.get("ok"):
#         return {"label": "unknown", "score": 0.0, "raw": dbg.get("json") if dbg else dbg, "source": "hf"}
#     out = dbg["json"]
#     if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
#         best = max(out, key=lambda x: x.get("score", 0))
#         return {"label": best.get("label"), "score": float(best.get("score", 0)), "raw": out, "source":"hf"}
#     if isinstance(out, dict):
#         if "label" in out and "score" in out:
#             return {"label": out["label"], "score": float(out["score"]), "raw": out, "source":"hf"}
#         if "labels" in out and "scores" in out and len(out["labels"])==len(out["scores"]) and len(out["labels"])>0:
#             idx = int(np.argmax(out["scores"]))
#             return {"label": out["labels"][idx], "score": float(out["scores"][idx]), "raw": out, "source":"hf"}
#         for v in out.values():
#             if isinstance(v, list) and len(v)>0 and isinstance(v[0], dict) and "label" in v[0]:
#                 best = max(v, key=lambda x: x.get("score",0))
#                 return {"label": best.get("label"), "score": float(best.get("score",0)), "raw": out, "source":"hf"}
#     return {"label":"unknown", "score":0.0, "raw": out, "source":"hf"}

# def aggregate_emotions(emotion_timeline):
#     """
#     emotion_timeline: list of {"time": float, "face_id": int, "emotion": str, "score": float}
#     returns dict: face_id -> {"top": (label,count), "avg_score": float, "samples": [times...], "raw_counts": {label:count}}
#     """
#     from collections import defaultdict, Counter
#     grouped = defaultdict(list)
#     for e in emotion_timeline:
#         fid = e.get("face_id", "unknown")
#         grouped[fid].append(e)

#     summary = {}
#     for fid, items in grouped.items():
#         labels = [it.get("emotion","unknown") for it in items]
#         scores = [float(it.get("score", 0.0) or 0.0) for it in items]
#         counts = Counter(labels)
#         top_label, top_count = counts.most_common(1)[0]
#         avg_score = sum(scores) / len(scores) if scores else 0.0
#         times = [round(float(it.get("time",0)), 2) for it in items][:10]  # sample up to 10 timestamps
#         summary[fid] = {
#             "top_label": top_label,
#             "top_count": int(top_count),
#             "samples": times,
#             "avg_score": round(avg_score, 3),
#             "total_samples": len(items),
#             "raw_counts": dict(counts)
#         }
#     return summary


# # ---------------------------
# # LLM analysis via REST with retry/backoff
# # ---------------------------
# def post_with_retry(url, headers, json_body, max_retries=6, base_delay=1.0, max_delay=30.0):
#     attempt = 0
#     while True:
#         attempt += 1
#         try:
#             resp = requests.post(url, headers=headers, json=json_body, timeout=120)
#         except requests.RequestException:
#             if attempt >= max_retries:
#                 raise
#             delay = min(max_delay, base_delay * (2 ** (attempt-1)))
#             delay = delay * (0.5 + random.random()*0.5)
#             time.sleep(delay)
#             continue
#         if resp.status_code == 200:
#             return resp
#         if resp.status_code == 429:
#             retry_after = resp.headers.get("Retry-After")
#             if retry_after:
#                 try:
#                     wait = float(retry_after)
#                 except Exception:
#                     wait = base_delay * (2 ** (attempt-1))
#             else:
#                 wait = min(max_delay, base_delay * (2 ** (attempt-1)))
#             wait = wait * (0.8 + 0.4 * random.random())
#             if attempt >= max_retries:
#                 resp.raise_for_status()
#             time.sleep(wait)
#             continue
#         if 500 <= resp.status_code < 600:
#             if attempt >= max_retries:
#                 resp.raise_for_status()
#             delay = min(max_delay, base_delay * (2 ** (attempt-1)))
#             delay = delay * (0.5 + random.random()*0.5)
#             time.sleep(delay)
#             continue
#         resp.raise_for_status()

# def call_openai_analysis_rest(transcript, segments, audio_events, emotion_timeline, speaker_labels=None,
#                               model="gpt-3.5-turbo", max_tokens=800, system_prompt=None):
#     MAX_CHARS = 30000
#     if transcript and len(transcript) > MAX_CHARS:
#         head = transcript[:10000]
#         mid = transcript[len(transcript)//2 : len(transcript)//2 + 5000]
#         tail = transcript[-10000:]
#         transcript_short = head + "\n\n...(middle omitted for brevity)...\n\n" + mid + "\n\n...omitted...\n\n" + tail
#     else:
#         transcript_short = transcript

#     if system_prompt is None:
#         system_prompt = (
#             "You are an expert educational coach. Given transcript segments, audio events (silence/clipping), "
#             "and an emotion summary per participant, produce a clear, human-friendly lesson report.\n\n"
#             "Required output (respond in plain Markdown):\n"
#             "1) Short 2-sentence SUMMARY of the lesson content.\n"
#             "2) KEY MOMENTS: a bulleted timeline of up to 5 important timestamps (timestamp — 1 short sentence).\n"
#             "3) EMOTION INSIGHTS: For each participant (labelled by id), write one short sentence describing the "
#             "dominant emotion, confidence/strength, and a one-line interpretation (what the teacher could infer).\n"
#             "   - Example: \"Person 7 — Mostly 'fear' (3/4 samples, avg score 0.78): may be anxious about the topic; consider checking in privately.\"\n"
#             "4) OVERALL ENGAGEMENT: write 1–2 sentences comparing teacher vs students engagement and what to improve.\n"
#             "5) AUDIO ISSUES: list any detected silences/dropouts with timestamps.\n"
#             "6) ACTION ITEMS: provide 3 prioritized, concrete teacher actions (each 1 line).\n\n"
#             "Use the provided 'emotion_summary' field (aggregated per face) to produce the EMOTION INSIGHTS. "
#             "Return the whole report in readable Markdown only (no JSON)."
#         )

#     emotion_summary = aggregate_emotions(emotion_timeline)

#     user_payload = {
#         "transcript": transcript_short,
#         "segments": segments,
#         "audio_events": audio_events,
#         #"emotion_timeline": emotion_timeline,
#         "emotion_summary": emotion_summary, 
#         "speaker_labels": speaker_labels
#     }

#     url = f"{OPENAI_BASE}/chat/completions"
#     headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
#     body = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": json.dumps(user_payload)}
#         ],
#         "temperature": 0.0,
#         "max_tokens": max_tokens,
#         "n": 1
#     }

#     try:
#         resp = post_with_retry(url, headers, body, max_retries=6)
#     except Exception as e:
#         return {"error": f"Request failed after retries: {e}"}

#     try:
#         data = resp.json()
#     except Exception as e:
#         return {"error": f"Invalid JSON response from OpenAI: {e}, raw: {resp.text[:1000]}"}

#     try:
#         text = data["choices"][0]["message"]["content"]
#     except Exception:
#         return {"error": "Unexpected response shape from OpenAI", "raw": data}

#     try:
#         parsed = json.loads(text)
#         return parsed
#     except Exception:
#         return {"raw_output": text, "raw_response": data}

# # ---------------------------
# # STREAMLIT UI
# # ---------------------------
# st.set_page_config(page_title="InsightEd MVP", layout="wide")
# st.title("InsightEd — MVP: AI-powered processing for online sessions")

# ffmpeg_bin = check_ffmpeg_available()
# if not ffmpeg_bin:
#     st.warning("ffmpeg not found in PATH. Install ffmpeg (conda install -c conda-forge ffmpeg or brew install ffmpeg) and restart.")

# st.markdown("**Privacy note:** ensure you have consent from all participants before uploading and processing recordings.")

# uploaded = st.file_uploader("Upload an MP4 video of an online session", type=["mp4", "mov", "mkv"], accept_multiple_files=False)

# sample_rate_sec = st.slider(
#     "Frame sampling rate (seconds)",
#     min_value=0.5,
#     max_value=5.0,
#     value=2.0,
#     step=0.5,
#     help="How often frames are analyzed. Lower = more frames = more detailed but slower. Higher = faster but less detailed."
# )

# run_emotion = st.checkbox("Run face emotion detection (Hugging Face)", value=True if HF_API_KEY else False)

# # prevent double runs
# if "running" not in st.session_state:
#     st.session_state["running"] = False

# if uploaded:
#     tmpdir = tempfile.mkdtemp()
#     video_path = os.path.join(tmpdir, uploaded.name)
#     save_uploaded_file(uploaded, video_path)
#     st.success("Saved. Click 'Run analysis' to start audio extraction and analysis.")

#     run_disabled = st.session_state["running"]
#     if st.button("Run analysis", disabled=run_disabled):
#         st.session_state["running"] = True
#         try:
#             # Extract audio
#             with st.spinner("Extracting audio..."):
#                 audio_path = os.path.join(tmpdir, "audio.wav")
#                 try:
#                     extract_audio_from_video(video_path, audio_path)
#                 except Exception as e:
#                     st.error(f"Audio extraction failed: {e}")
#                     st.session_state["running"] = False
#                     st.stop()

#             # Transcribe
#             with st.spinner("Transcribing audio..."):
#                 transcription = transcribe_with_whisper_rest(audio_path)

#             # Audio events
#             with st.spinner("Analyzing audio quality..."):
#                 silences = detect_silences(audio_path, min_silence_len_ms=800, silence_thresh_db=-40)
#                 audio_events = [{"type":"silence", "start": s, "end": e} for s,e in silences]

#             # Emotions
#             emotion_timeline = []
#             if run_emotion and HF_API_KEY:
#                 with st.spinner("Sampling frames and detecting faces..."):
#                     frames = sample_frames(video_path, sample_rate_sec=sample_rate_sec)
#                     frames_with_faces = []
#                     for t, frame in frames:
#                         faces = detect_faces_in_frame(frame)
#                         faces_list = []
#                         for (x,y,w,h) in faces:
#                             faces_list.append((x,y,w,h))
#                         frames_with_faces.append((t, faces_list))
#                     tracked = track_faces(frames_with_faces)

#                     max_calls_per_face = 3
#                     calls_per_face = defaultdict(int)

#                     for t, boxes in tracked:
#                         # find the nearest frame image to time t
#                         frame_img = None
#                         for ft, frm in frames:
#                             if abs(ft - t) < 0.05:
#                                 frame_img = frm
#                                 break
#                         # explicit None check (do NOT use `or` with numpy arrays)
#                         if frame_img is None:
#                             frame_img = frames[0][1] if len(frames) > 0 else None

#                         if frame_img is None:
#                             continue

#                         for tid, box in boxes:
#                             if calls_per_face[tid] >= max_calls_per_face:
#                                 continue
#                             x,y,w,h = box
#                             h_frame, w_frame = frame_img.shape[:2]
#                             x0 = max(0, int(x)); y0 = max(0, int(y))
#                             x1 = min(w_frame, int(x + w)); y1 = min(h_frame, int(y + h))
#                             if x1 <= x0 or y1 <= y0:
#                                 continue
#                             crop_img = frame_img[y0:y1, x0:x1]

#                             if crop_img.shape[1] < 48 or crop_img.shape[0] < 48:
#                                 continue
#                             if crop_img.mean() < 10:
#                                 continue
#                             res = classify_face_emotion(crop_img, HF_API_KEY, HF_EMOTION_MODEL)
#                             # low confidence handling
#                             label = res.get("label","unknown")
#                             score = float(res.get("score", 0.0) or 0.0)
#                             if score < 0.4:
#                                 label = "unknown (low confidence)"
#                             calls_per_face[tid] += 1
#                             emotion_timeline.append({"time": t, "face_id": tid, "emotion": label, "score": score})
#             else:
#                 emotion_timeline = []

#             # LLM analysis
#             with st.spinner("Generating AI report..."):
#                 report = call_openai_analysis_rest(transcription.get("text",""), transcription.get("segments", []), audio_events, emotion_timeline, speaker_labels=None)

#             # ---------- UI: show results ----------
#             st.header("AI Report")
#             r = report  # ответ от call_openai_analysis_rest

#             # Normalize: если модель вернула raw_output (строка), упакуем его аккуратно
#             final_report = r
#             if isinstance(r, dict) and "raw_output" in r and isinstance(r["raw_output"], str):
#                 raw_out = r["raw_output"].strip()
#                 # Попытка: если похоже на JSON, распарсить
#                 if (raw_out.startswith("{") and raw_out.endswith("}")) or (raw_out.startswith("[") and raw_out.endswith("]")):
#                     try:
#                         final_report = json.loads(raw_out)
#                     except Exception:
#                         final_report = {"_raw_markdown_or_text": raw_out, "raw_response": r.get("raw_response")}
#                 else:
#                     # иначе считаем это markdown/текстом
#                     final_report = {"_raw_markdown_or_text": raw_out, "raw_response": r.get("raw_response")}

#             # Case A: структурированный отчет (summary/key_moments/engagement/etc.)
#             if isinstance(final_report, dict) and any(k in final_report for k in ("summary", "key_moments", "engagement", "audio_issues", "actions")):
#                 st.subheader("Summary")
#                 st.write(final_report.get("summary", ""))

#                 st.subheader("Key moments")
#                 for km in final_report.get("key_moments", []):
#                     ts = km.get("timestamp", "")
#                     note = km.get("note", "")
#                     st.write(f"- **{ts}** — {note}")

#                 st.subheader("Engagement")
#                 for person, info in final_report.get("engagement", {}).items():
#                     st.write(f"- **{person}** — emotion: {info.get('emotion','')}, engagement: {info.get('engagement_level','')}")

#                 st.subheader("Audio issues")
#                 st.write(final_report.get("audio_issues", []))

#                 st.subheader("Action items")
#                 for i, act in enumerate(final_report.get("actions", []), start=1):
#                     if isinstance(act, dict):
#                         st.write(f"{i}. {act.get('action') or act.get('text') or str(act)}")
#                     else:
#                         st.write(f"{i}. {act}")

#                 # single structured download
#                 st.download_button("Download structured JSON", json.dumps(final_report, indent=2), file_name="insighted_report.json", key="download_structured")

#             # Case B: Markdown / plain text returned by the model (we render it nicely)
#             elif isinstance(final_report, dict) and "_raw_markdown_or_text" in final_report:
#                 md = final_report["_raw_markdown_or_text"]
#                 # Render as markdown (safe); если не получится — fallback на обычный write
#                 try:
#                     st.markdown(md, unsafe_allow_html=False)
#                 except Exception:
#                     st.write(md)
#                 st.download_button("Download report (MD/TXT)", md, file_name="insighted_report.md", key="download_md")

#             # Fallback: неизвестный формат — показываем аккуратно в UI и даём скачивание
#             else:
#                 try:
#                     st.json(final_report)
#                     st.download_button("Download raw report", json.dumps(final_report, indent=2), file_name="insighted_report_raw.json", key="download_raw")
#                 except Exception:
#                     st.write(str(final_report))
#                     st.download_button("Download raw report", str(final_report), file_name="insighted_report_raw.txt", key="download_raw_text")

#             # Raw API response & transcript in an expander (for power users / debugging)
#             with st.expander("Transcript and raw API response (open to view)"):
#                 st.subheader("Transcript")
#                 st.write(transcription.get("text", ""))

#                 st.subheader("Raw API response (full)")
#                 raw_resp = None
#                 if isinstance(r, dict) and "raw_response" in r:
#                     raw_resp = r["raw_response"]
#                 elif isinstance(final_report, dict) and "raw_response" in final_report:
#                     raw_resp = final_report["raw_response"]

#                 if raw_resp:
#                     try:
#                         if isinstance(raw_resp, dict):
#                             meta = {k: raw_resp.get(k) for k in ("id", "model", "usage")}
#                             st.write(meta)
#                             choices = raw_resp.get("choices")
#                             if choices and isinstance(choices, list) and len(choices) > 0:
#                                 content = choices[0].get("message", {}).get("content", "")
#                                 st.code(content[:4000])
#                             else:
#                                 st.json(raw_resp)
#                         else:
#                             st.code(str(raw_resp)[:4000])
#                     except Exception:
#                         st.write(str(raw_resp)[:4000])
#                 else:
#                     st.write("No raw API response available.")
                         
#         finally:
#             st.session_state["running"] = False

# app.py (clean, production-friendly MVP)
import os
import io
import time
import json
import tempfile
import requests
import numpy as np
import streamlit as st
from pydub import AudioSegment, silence
import cv2
import shutil
import subprocess
import random
import hashlib
from collections import defaultdict, Counter

# ---------------------------
# CONFIG / SECRETS
# ---------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
HF_API_KEY = st.secrets.get("HF_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("OpenAI API key missing in .streamlit/secrets.toml or environment.")
    st.stop()

OPENAI_BASE = "https://api.openai.com/v1"
HEADERS_AUTH = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# ---------------------------
# UTIL: file helpers
# ---------------------------
def save_uploaded_file(uploaded_file, dst_path):
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst_path

# ---------------------------
# AUDIO / VIDEO UTIL (ffmpeg CLI)
# ---------------------------
# Попробуй найти системный ffmpeg, иначе используем imageio-ffmpeg
def check_ffmpeg_available():
    """Return path to ffmpeg if available. Try system ffmpeg first, then imageio-ffmpeg."""
    # 1) system ffmpeg on PATH
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path

    # 2) try imageio-ffmpeg package binary
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        exe = get_ffmpeg_exe()  # returns path to bundled ffmpeg binary
        if exe:
            return exe
    except Exception:
        pass

    # not found
    return None


def extract_audio_from_video(video_path, out_audio_path, sample_rate=16000):
    ffmpeg_path = check_ffmpeg_available()
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is not found in PATH. Please install system ffmpeg.\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu/Debian: sudo apt install ffmpeg\n"
            "Windows: install ffmpeg and add to PATH"
        )
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        out_audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not (os.path.exists(out_audio_path) and os.path.getsize(out_audio_path) > 0):
        raise RuntimeError("Audio extraction completed but output file is missing or empty.")
    return out_audio_path

def detect_silences(audio_path, min_silence_len_ms=1000, silence_thresh_db=-40):
    audio = AudioSegment.from_wav(audio_path)
    silences = silence.detect_silence(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
    silences_sec = [(s/1000.0, e/1000.0) for s,e in silences]
    return silences_sec

# ---------------------------
# Transcription via REST (Whisper)
# ---------------------------
def transcribe_with_whisper_rest(audio_path):
    url = f"{OPENAI_BASE}/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "whisper-1"}
        headers = HEADERS_AUTH.copy()
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            st.error(f"Whisper transcription error: {e}")
            return {"text": "", "segments": []}
    text = j.get("text", "") if isinstance(j, dict) else ""
    segments = j.get("segments", []) if isinstance(j, dict) else []
    return {"text": text, "segments": segments}

# ---------------------------
# FRAME SAMPLING & FACE DETECTION
# ---------------------------
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def sample_frames(video_path, sample_rate_sec=2.0, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file for frame sampling.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    frames = []
    if duration == 0:
        while len(frames) < max_frames:
            success, frame = cap.read()
            if not success:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append((float(t), frame.copy()))
        cap.release()
        return frames
    times = np.arange(0, duration, sample_rate_sec)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
        success, frame = cap.read()
        if not success:
            continue
        frames.append((float(t), frame.copy()))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

def detect_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_XML)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
    return faces

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0
    return interArea / union

def track_faces(frames_with_faces, iou_thresh=0.3):
    tracks = {}
    next_id = 1
    timeline = []
    for t, boxes in frames_with_faces:
        row = []
        for box in boxes:
            assigned = False
            for tid, last_box in tracks.items():
                if iou(box, last_box) > iou_thresh:
                    tracks[tid] = box
                    row.append((tid, box))
                    assigned = True
                    break
            if not assigned:
                tracks[next_id] = box
                row.append((next_id, box))
                next_id += 1
        timeline.append((t, row))
    return timeline

# ---------------------------
# HUGGINGFACE EMOTION INFERENCE (cache + parser)
# ---------------------------
HF_EMOTION_MODEL = "dima806/facial_emotions_image_detection"
_hf_cache = {}

def hf_classify_image_raw_jpeg(model_name, image_bytes_io, hf_token, timeout=60):
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    headers["Content-Type"] = "image/jpeg"
    try:
        resp = requests.post(api_url, headers=headers, data=image_bytes_io.getvalue(), timeout=timeout)
    except requests.exceptions.Timeout as e:
        return {"ok": False, "status_code": None, "text": "", "json": None, "error": f"timeout: {e}"}
    except Exception as e:
        return {"ok": False, "status_code": None, "text": "", "json": None, "error": f"request error: {e}"}
    text = resp.text[:4000]
    try:
        parsed = resp.json()
    except Exception:
        parsed = None
    if resp.status_code != 200:
        return {"ok": False, "status_code": resp.status_code, "text": text, "json": parsed, "error": None}
    return {"ok": True, "status_code": resp.status_code, "text": text, "json": parsed, "error": None}

def hf_infer_with_cache(model_name, img_bytes_io, hf_token):
    key = hashlib.sha256(img_bytes_io.getvalue()).hexdigest()
    cache_key = f"{model_name}:{key}"
    if cache_key in _hf_cache:
        return _hf_cache[cache_key]
    dbg = hf_classify_image_raw_jpeg(model_name, img_bytes_io, hf_token)
    _hf_cache[cache_key] = dbg
    return dbg

def classify_face_emotion(crop_bgr, hf_token, model_name):
    _, buf = cv2.imencode('.jpg', crop_bgr)
    img_bytes = io.BytesIO(buf.tobytes())
    if not hf_token:
        return {"label": "unknown", "score": 0.0, "raw": None, "source": "none"}
    dbg = hf_infer_with_cache(model_name, img_bytes, hf_token)
    if not dbg or not dbg.get("ok"):
        return {"label": "unknown", "score": 0.0, "raw": dbg.get("json") if dbg else dbg, "source": "hf"}
    out = dbg["json"]
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        best = max(out, key=lambda x: x.get("score", 0))
        return {"label": best.get("label"), "score": float(best.get("score", 0)), "raw": out, "source":"hf"}
    if isinstance(out, dict):
        if "label" in out and "score" in out:
            return {"label": out["label"], "score": float(out["score"]), "raw": out, "source":"hf"}
        if "labels" in out and "scores" in out and len(out["labels"])==len(out["scores"]) and len(out["labels"])>0:
            idx = int(np.argmax(out["scores"]))
            return {"label": out["labels"][idx], "score": float(out["scores"][idx]), "raw": out, "source":"hf"}
        for v in out.values():
            if isinstance(v, list) and len(v)>0 and isinstance(v[0], dict) and "label" in v[0]:
                best = max(v, key=lambda x: x.get("score",0))
                return {"label": best.get("label"), "score": float(best.get("score",0)), "raw": out, "source":"hf"}
    return {"label":"unknown", "score":0.0, "raw": out, "source":"hf"}

def aggregate_emotions(emotion_timeline):
    """
    emotion_timeline: list of {"time": float, "face_id": int, "emotion": str, "score": float}
    returns dict: face_id -> {"top": (label,count), "avg_score": float, "samples": [times...], "raw_counts": {label:count}}
    """
    from collections import defaultdict, Counter
    grouped = defaultdict(list)
    for e in emotion_timeline:
        fid = e.get("face_id", "unknown")
        grouped[fid].append(e)

    summary = {}
    for fid, items in grouped.items():
        labels = [it.get("emotion","unknown") for it in items]
        scores = [float(it.get("score", 0.0) or 0.0) for it in items]
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        times = [round(float(it.get("time",0)), 2) for it in items][:10]  # sample up to 10 timestamps
        summary[fid] = {
            "top_label": top_label,
            "top_count": int(top_count),
            "samples": times,
            "avg_score": round(avg_score, 3),
            "total_samples": len(items),
            "raw_counts": dict(counts)
        }
    return summary


# ---------------------------
# LLM analysis via REST with retry/backoff
# ---------------------------
def post_with_retry(url, headers, json_body, max_retries=6, base_delay=1.0, max_delay=30.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=120)
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt-1)))
            delay = delay * (0.5 + random.random()*0.5)
            time.sleep(delay)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except Exception:
                    wait = base_delay * (2 ** (attempt-1))
            else:
                wait = min(max_delay, base_delay * (2 ** (attempt-1)))
            wait = wait * (0.8 + 0.4 * random.random())
            if attempt >= max_retries:
                resp.raise_for_status()
            time.sleep(wait)
            continue
        if 500 <= resp.status_code < 600:
            if attempt >= max_retries:
                resp.raise_for_status()
            delay = min(max_delay, base_delay * (2 ** (attempt-1)))
            delay = delay * (0.5 + random.random()*0.5)
            time.sleep(delay)
            continue
        resp.raise_for_status()

def call_openai_analysis_rest(transcript, segments, audio_events, emotion_timeline, speaker_labels=None,
                              model="gpt-3.5-turbo", max_tokens=800, system_prompt=None):
    MAX_CHARS = 30000
    if transcript and len(transcript) > MAX_CHARS:
        head = transcript[:10000]
        mid = transcript[len(transcript)//2 : len(transcript)//2 + 5000]
        tail = transcript[-10000:]
        transcript_short = head + "\n\n...(middle omitted for brevity)...\n\n" + mid + "\n\n...omitted...\n\n" + tail
    else:
        transcript_short = transcript

    if system_prompt is None:
        system_prompt = (
            "You are an expert educational coach. Given transcript segments, audio events (silence/clipping), "
            "and an emotion summary per participant, produce a clear, human-friendly lesson report.\n\n"
            "Required output (respond in plain Markdown):\n"
            "1) Short 2-sentence SUMMARY of the lesson content.\n"
            "2) KEY MOMENTS: a bulleted timeline of up to 5 important timestamps (timestamp — 1 short sentence).\n"
            "3) EMOTION INSIGHTS: For each participant (labelled by id), write one short sentence describing the "
            "dominant emotion, confidence/strength, and a one-line interpretation (what the teacher could infer).\n"
            "   - Example: \"Person 7 — Mostly 'fear' (3/4 samples, avg score 0.78): may be anxious about the topic; consider checking in privately.\"\n"
            "4) OVERALL ENGAGEMENT: write 1–2 sentences comparing teacher vs students engagement and what to improve.\n"
            "5) AUDIO ISSUES: list any detected silences/dropouts with timestamps.\n"
            "6) ACTION ITEMS: provide 3 prioritized, concrete teacher actions (each 1 line).\n\n"
            "Use the provided 'emotion_summary' field (aggregated per face) to produce the EMOTION INSIGHTS. "
            "Return the whole report in readable Markdown only (no JSON)."
        )

    emotion_summary = aggregate_emotions(emotion_timeline)

    user_payload = {
        "transcript": transcript_short,
        "segments": segments,
        "audio_events": audio_events,
        "emotion_summary": emotion_summary,
        "speaker_labels": speaker_labels
    }

    url = f"{OPENAI_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "n": 1
    }

    try:
        resp = post_with_retry(url, headers, body, max_retries=6)
    except Exception as e:
        return {"error": f"Request failed after retries: {e}"}

    try:
        data = resp.json()
    except Exception as e:
        return {"error": f"Invalid JSON response from OpenAI: {e}, raw: {resp.text[:1000]}"}

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        return {"error": "Unexpected response shape from OpenAI", "raw": data}

    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return {"raw_output": text, "raw_response": data}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="InsightEd MVP", layout="wide")
st.title("InsightEd — MVP: AI-powered processing for online sessions")

ffmpeg_bin = check_ffmpeg_available()
if not ffmpeg_bin:
    st.warning("ffmpeg not found in PATH. Install ffmpeg (conda install -c conda-forge ffmpeg or brew install ffmpeg) and restart.")

st.markdown("**Privacy note:** ensure you have consent from all participants before uploading and processing recordings.")

# Allow either upload or example video
uploaded = st.file_uploader("Upload an MP4 video of an online session", type=["mp4", "mov", "mkv"], accept_multiple_files=False)

# Example video button (if file exists in repo root)
example_path = "example.mov"
use_example = False
if os.path.exists(example_path):
    if st.button("Use example video"):
        st.session_state["use_example"] = True
    # allow user to clear example selection
    if st.session_state.get("use_example"):
        use_example = True
        st.info("Using example video from project folder.")
        # show the example video preview
        try:
            st.video(example_path)
        except Exception:
            st.write("Example video available (preview not supported in this environment).")

# If user uploaded file, show preview
if uploaded:
    try:
        st.video(uploaded)
    except Exception:
        st.write("Uploaded video (preview not available).")

# Frame sampling slider
sample_rate_sec = st.slider(
    "Frame sampling rate (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="How often frames are analyzed. Lower = more frames = more detailed but slower. Higher = faster but less detailed."
)

run_emotion = st.checkbox("Run face emotion detection (Hugging Face)", value=True if HF_API_KEY else False)

# prevent double runs
if "running" not in st.session_state:
    st.session_state["running"] = False

# initialize storage for results if missing
if "last_report" not in st.session_state:
    st.session_state["last_report"] = None
if "last_transcription" not in st.session_state:
    st.session_state["last_transcription"] = None
if "last_audio_events" not in st.session_state:
    st.session_state["last_audio_events"] = None
if "last_emotion_timeline" not in st.session_state:
    st.session_state["last_emotion_timeline"] = None
if "last_raw_response" not in st.session_state:
    st.session_state["last_raw_response"] = None
if "last_video_path" not in st.session_state:
    st.session_state["last_video_path"] = None

# Determine active video path (uploaded or example)
video_selected = False
video_path = None
tmpdir = None
if use_example:
    video_path = example_path
    st.session_state["last_video_path"] = video_path
    video_selected = True
elif uploaded:
    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, uploaded.name)
    save_uploaded_file(uploaded, video_path)
    st.session_state["last_video_path"] = video_path
    video_selected = True
else:
    # if session has previous video path (from prior run), keep it for preview
    if st.session_state.get("last_video_path"):
        prev = st.session_state["last_video_path"]
        if os.path.exists(prev):
            video_path = prev
            video_selected = True

# Show "Run analysis" button only when a video is selected
if not video_selected:
    st.info("Upload a video or click 'Use example video' to start analysis.")
else:
    run_disabled = st.session_state["running"]
    if st.button("Run analysis", disabled=run_disabled):
        st.session_state["running"] = True
        try:
            # Extract audio
            with st.spinner("Extracting audio..."):
                audio_path = os.path.join(tempfile.mkdtemp(), "audio.wav")
                try:
                    extract_audio_from_video(video_path, audio_path)
                except Exception as e:
                    st.error(f"Audio extraction failed: {e}")
                    st.session_state["running"] = False
                    st.stop()

            # Transcribe
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_with_whisper_rest(audio_path)

            # Audio events
            with st.spinner("Analyzing audio quality..."):
                silences = detect_silences(audio_path, min_silence_len_ms=800, silence_thresh_db=-40)
                audio_events = [{"type":"silence", "start": s, "end": e} for s,e in silences]

            # Emotions
            emotion_timeline = []
            if run_emotion and HF_API_KEY:
                with st.spinner("Sampling frames and detecting faces..."):
                    frames = sample_frames(video_path, sample_rate_sec=sample_rate_sec)
                    frames_with_faces = []
                    for t, frame in frames:
                        faces = detect_faces_in_frame(frame)
                        faces_list = []
                        for (x,y,w,h) in faces:
                            faces_list.append((x,y,w,h))
                        frames_with_faces.append((t, faces_list))
                    tracked = track_faces(frames_with_faces)

                    max_calls_per_face = 3
                    calls_per_face = defaultdict(int)

                    for t, boxes in tracked:
                        # find the nearest frame image to time t
                        frame_img = None
                        for ft, frm in frames:
                            if abs(ft - t) < 0.05:
                                frame_img = frm
                                break
                        # explicit None check (do NOT use `or` with numpy arrays)
                        if frame_img is None:
                            frame_img = frames[0][1] if len(frames) > 0 else None

                        if frame_img is None:
                            continue

                        for tid, box in boxes:
                            if calls_per_face[tid] >= max_calls_per_face:
                                continue
                            x,y,w,h = box
                            h_frame, w_frame = frame_img.shape[:2]
                            x0 = max(0, int(x)); y0 = max(0, int(y))
                            x1 = min(w_frame, int(x + w)); y1 = min(h_frame, int(y + h))
                            if x1 <= x0 or y1 <= y0:
                                continue
                            crop_img = frame_img[y0:y1, x0:x1]

                            if crop_img.shape[1] < 48 or crop_img.shape[0] < 48:
                                continue
                            if crop_img.mean() < 10:
                                continue
                            res = classify_face_emotion(crop_img, HF_API_KEY, HF_EMOTION_MODEL)
                            # low confidence handling
                            label = res.get("label","unknown")
                            score = float(res.get("score", 0.0) or 0.0)
                            if score < 0.4:
                                label = "unknown (low confidence)"
                            calls_per_face[tid] += 1
                            emotion_timeline.append({"time": t, "face_id": tid, "emotion": label, "score": score})
            else:
                emotion_timeline = []

            # LLM analysis
            with st.spinner("Generating AI report..."):
                report = call_openai_analysis_rest(transcription.get("text",""), transcription.get("segments", []), audio_events, emotion_timeline, speaker_labels=None)

            # ---------- UI: show results ----------
            # Normalize: если модель вернула raw_output (строка), упакуем его аккуратно
            r = report  # ответ от call_openai_analysis_rest
            final_report = r
            raw_resp_for_save = None
            if isinstance(r, dict) and "raw_output" in r and isinstance(r["raw_output"], str):
                raw_out = r["raw_output"].strip()
                # Попытка: если похоже на JSON, распарсить
                if (raw_out.startswith("{") and raw_out.endswith("}")) or (raw_out.startswith("[") and raw_out.endswith("]")):
                    try:
                        final_report = json.loads(raw_out)
                    except Exception:
                        final_report = {"_raw_markdown_or_text": raw_out, "raw_response": r.get("raw_response")}
                else:
                    # иначе считаем это markdown/текстом
                    final_report = {"_raw_markdown_or_text": raw_out, "raw_response": r.get("raw_response")}
                raw_resp_for_save = r.get("raw_response")
            else:
                # if full raw_response present, capture for saving
                if isinstance(r, dict) and "raw_response" in r:
                    raw_resp_for_save = r.get("raw_response")
                else:
                    raw_resp_for_save = r

            # Save results to session so they persist across interactions
            st.session_state["last_report"] = final_report
            st.session_state["last_transcription"] = transcription
            st.session_state["last_audio_events"] = audio_events
            st.session_state["last_emotion_timeline"] = emotion_timeline
            st.session_state["last_raw_response"] = raw_resp_for_save
            st.session_state["last_video_path"] = video_path

            # Display the report immediately (also persistent)
            st.success("Analysis completed and saved in session.")
        finally:
            st.session_state["running"] = False

# ---------------------------
# Display saved/persisted results (if any)
# ---------------------------
if st.session_state.get("last_report") is not None:
    st.header("AI Report")
    final_report = st.session_state["last_report"]
    transcription = st.session_state.get("last_transcription") or {"text": ""}
    audio_events = st.session_state.get("last_audio_events") or []
    emotion_timeline = st.session_state.get("last_emotion_timeline") or []
    raw_resp = st.session_state.get("last_raw_response")

    # Case A: structured dict with expected keys (summary, key_moments, etc.)
    if isinstance(final_report, dict) and any(k in final_report for k in ("summary","key_moments","engagement","audio_issues","actions")):
        st.subheader("Summary")
        st.write(final_report.get("summary", ""))

        st.subheader("Key moments")
        for km in final_report.get("key_moments", []):
            ts = km.get("timestamp", "")
            note = km.get("note","")
            st.write(f"- **{ts}** — {note}")

        st.subheader("Engagement")
        for person, info in final_report.get("engagement", {}).items():
            st.write(f"- **{person}** — emotion: {info.get('emotion','')}, engagement: {info.get('engagement_level','')}")

        st.subheader("Audio issues")
        st.write(final_report.get("audio_issues", []))

        st.subheader("Action items")
        for i, act in enumerate(final_report.get("actions", []), start=1):
            if isinstance(act, dict):
                st.write(f"{i}. {act.get('action') or act.get('text') or str(act)}")
            else:
                st.write(f"{i}. {act}")

        # Download structured JSON enriched with transcript and raw response
        combined = {
            "report": final_report,
            "transcript": transcription.get("text",""),
            "audio_events": audio_events,
            "emotion_timeline": emotion_timeline,
            "raw_api_response": raw_resp
        }
        st.download_button("Download combined JSON report (report + transcript + raw API)", json.dumps(combined, indent=2), file_name="insighted_report_combined.json", key="download_combined")
    else:
        # If the final_report is plain markdown text inside dict key
        if isinstance(final_report, dict) and "_raw_markdown_or_text" in final_report:
            md = final_report["_raw_markdown_or_text"]
            try:
                st.markdown(md, unsafe_allow_html=False)
            except Exception:
                st.write(md)
            # Compose a combined markdown file including transcript and raw response
            md_combined = md + "\n\n---\n\n## Transcript\n\n" + (transcription.get("text","") or "") + "\n\n---\n\n## Raw API response\n\n"
            md_combined += json.dumps(raw_resp, indent=2) if raw_resp else "No raw response."
            st.download_button("Download combined MD report (MD + transcript + raw API)", md_combined, file_name="insighted_report.md", key="download_combined_md")
        else:
            # fallback: show raw
            try:
                st.json(final_report)
            except Exception:
                st.write(str(final_report))
            combined = {
                "report": final_report,
                "transcript": transcription.get("text",""),
                "audio_events": audio_events,
                "emotion_timeline": emotion_timeline,
                "raw_api_response": raw_resp
            }
            st.download_button("Download combined JSON report (report + transcript + raw API)", json.dumps(combined, indent=2), file_name="insighted_report_combined.json", key="download_combined_fallback")

    # Hidden raw details (expandable)
    with st.expander("Transcript and raw API response (open to view)"):
        st.subheader("Transcript")
        st.write(transcription.get("text",""))
        st.subheader("Audio events")
        if audio_events:
            rows = [{"type": ev["type"], "start_s": round(ev["start"],3), "end_s": round(ev["end"],3), "dur_s": round(ev["end"]-ev["start"],3)} for ev in audio_events]
            st.table(rows)
        else:
            st.write("No audio events detected.")
        st.subheader("Emotion timeline (sampled)")
        if emotion_timeline:
            grouped = defaultdict(list)
            for e in emotion_timeline:
                grouped[e["face_id"]].append(e)
            for fid, items in grouped.items():
                st.write(f"Person {fid} — samples: {len(items)}")
                simple_tl = [f"{int(it['time'])}s: {it['emotion']} ({round(it.get('score',0),2)})" for it in items[:30]]
                st.write(", ".join(simple_tl))
        else:
            st.write("No emotion data.")

        st.subheader("Raw API response (truncated)")
        if raw_resp:
            try:
                if isinstance(raw_resp, dict):
                    meta = {k: raw_resp.get(k) for k in ("id","model","usage")}
                    st.write(meta)
                    choices = raw_resp.get("choices")
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        st.code(content[:4000])
                    else:
                        st.json(raw_resp)
                else:
                    st.code(str(raw_resp)[:4000])
            except Exception:
                st.write(str(raw_resp)[:4000])
        else:
            st.write("No raw API response available.")
