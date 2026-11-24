"""Download the MAV-Celeb v2 release (organisers' public Google Drive) with retries (anonymous Drive downloads are rate limited).
    python src/data_prep/download_mavceleb_v2.py [target directory]
"""
import os, sys, time
import gdown
D = sys.argv[1] if len(sys.argv) > 1 else "mavceleb_download"; os.makedirs(D, exist_ok=True)
FILE_ID, NAME = "1OY2x3G6JnA7g779CADhA3FCCrrrL0gSE", "mavceleb_v2_complete.zip"
out = os.path.join(D, NAME)
for attempt in range(300):
    if os.path.exists(out): break
    try:
        if gdown.download(id=FILE_ID, output=out + ".part", quiet=True, resume=True) and os.path.exists(out + ".part"):
            os.replace(out + ".part", out)
    except Exception as e:
        print(time.strftime("%H:%M"), "retry:", repr(e)[:90], flush=True); time.sleep(240)
print("DONE", out, os.path.getsize(out), flush=True)
