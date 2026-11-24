"""Downloads every public pretrained model the final system uses (network access needed; run once).
Hugging Face models go to $HF_HOME, torch.hub models to $TORCH_HOME, InsightFace to $INSIGHTFACE_ROOT (default ~/.insightface)."""
import os
from huggingface_hub import snapshot_download
for mid in ("openai/clip-vit-large-patch14",                       # face: CLIP ViT-L/14
            "microsoft/wavlm-base-plus",                           # voice: WavLM-base-plus
            "audeering/wav2vec2-large-robust-24-ft-age-gender",    # voice: age/gender wav2vec2
            "speechbrain/spkrec-ecapa-voxceleb",                   # voice: ECAPA-TDNN
            "Wespeaker/wespeaker-voxceleb-resnet293-LM"):           # voice: WeSpeaker ResNet293-LM (ONNX)
    print("OK", mid, snapshot_download(mid), flush=True)
import timm, torch
timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True)            # face: PE-Core-L/14-336
for name, tt, ds in (("b6", "ft_lm", "vox2"), ("M", "ft_mix", "vb2+vox2+cnc")):    # voice: ReDimNet-B6, ReDimNet-M
    torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name=name, train_type=tt, dataset=ds, trust_repo=True)
from insightface.app import FaceAnalysis                                            # anchor term: ArcFace (buffalo_l)   
FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface")))
print("ALL MODELS DOWNLOADED", flush=True)
