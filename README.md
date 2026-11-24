# FLAG 2027: Face-Voice Association with Frozen Encoders, a Two-Tower Ensemble and an Anchor Consistency Term

Code, configurations and trained models of our system for the **FLAG 2027 Grand Challenge**
(Face-voice Association across LAnguages and Gender, ICASSP 2027).

Given **one face image and one voice clip**, the system outputs a distance; a lower value means the face and the voice
belong to the same person. It is evaluated in four conditions: English (a language seen in training) and Bangla (a
language never seen in training), each with unconstrained negatives and with same-gender negatives.

## Results (evaluation phase, EER %)

| No-gender English | No-gender Bangla | Gender English | Gender Bangla | **Overall** |
|:---:|:---:|:---:|:---:|:---:|
| 8.67 | 9.48 | 19.71 | 19.83 | **14.42** |

The four score files of the submission are reproduced **exactly** by `src/combine.py` from the stored per-trial scores, and the stored
scores are reproduced from the released checkpoints by `src/infer.py --check`.

---

## Contents

1. [Method at a glance](#method-at-a-glance)
2. [Method in detail](#method-in-detail)
3. [Compliance with the challenge rules](#compliance-with-the-challenge-rules)
4. [Repository structure](#repository-structure)
5. [Installation](#installation)
6. [Data](#data)
7. [Reproducing the results](#reproducing-the-results)
8. [Validation tools](#validation-tools)
9. [Pretrained models and licences](#pretrained-models-and-licences)

---

## Method at a glance

```
   face image                                    voice clip (16 kHz, first 10 s)
       │                                                 │
       ▼                                                 ▼
 ┌──────────────────────┐                  ┌───────────────────────────────────────┐
 │ frozen face encoders │                  │ frozen voice encoders                 │
 │ CLIP ViT-L/14        │                  │ WavLM · wav2vec2 age/gender · ECAPA   │
 │ PE-Core-L/14-336     │                  │ ReDimNet-B6 · ReDimNet-M · ResNet293  │
 └──────────┬───────────┘                  └───────────────────┬───────────────────┘
            │                                                  │
            ▼                                                  ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ TWO-TOWER BASE MODELS (the only trained part)                                      │
 │   face tower ─► 128-d ─┐                                                           │
 │                        ├─► cosine similarity                                       │
 │   voice tower ─► 128-d ┘                                                           │
 │ 9 models per condition (3 generations × 3 voice-input sets), 10 seeds each, mean   │
 │ + 4 gender-expert members in the two same-gender conditions                        │
 └─────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │ base score s
                                       │
 ┌─────────────────────────────────────┼──────────────────────────────────────────────┐
 │ ANCHOR CONSISTENCY TERM (no         │                                              │
 │ training): 7,405 anchor pairs from  │   face near an anchor? → voice vs its pair   │
 │ public corpora (mean ArcFace face / │   voice near an anchor? → face vs its pair   │
 │ mean ReDimNet-B6 voice)             │   no anchor close → contribution ≈ 0         │
 └─────────────────────────────────────┼──────────────────────────────────────────────┘
                                       ▼
                     final score = s + w · m     (w fixed per condition)
                     distance written to the submission = −final score
```

## Method in detail

### 1. Frozen encoders

All encoders are public checkpoints used without any training or adaptation. Every embedding is computed from one file
and L2-normalised; the voice embeddings of a model are concatenated into one voice vector.

| Modality | Encoder | Dim. | Used by |
|---|---|---|---|
| Face | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | 768 | base models |
| Face | PE-Core-L/14-336 (`timm/vit_pe_core_large_patch14_336.fb`) | 1024 | base models (generation 2) |
| Face | ArcFace R50, InsightFace `buffalo_l` | 512 | anchor consistency term |
| Voice | WavLM-base-plus, mean of all layers and frames | 768 | base models |
| Voice | wav2vec2-large-robust age/gender (audeering), mean of the last hidden state | 1024 | base models |
| Voice | ECAPA-TDNN (SpeechBrain, VoxCeleb) | 192 | base models |
| Voice | ReDimNet-B6 (VoxCeleb2) | 192 | base models, anchor consistency term |
| Voice | ReDimNet-M (VoxBlink2 + VoxCeleb2 + CN-Celeb) | 192 | base models |
| Voice | WeSpeaker ResNet293-LM (VoxCeleb) | 256 | base models (generations 2, 3) |

### 2. Two-tower base model

Each tower is `Dropout(0.1) → Linear(1024) → BatchNorm → ReLU → Dropout(p) → Linear(128) → L2 norm`; the score is the
cosine between the face and voice embeddings (`src/model.py`).

**Training.** Each batch holds 256 identities with 2 face images and 2 voice clips each, drawn independently, so a face
and a voice of the same person usually come from different recordings. The loss is a symmetric multi-positive InfoNCE
over the 512 × 512 similarity matrix with a fixed temperature: all faces and voices of the same identity are positives,
everything else in the batch is a negative. Further details:

- the 70 challenge training identities are sampled 20 times more often than external identities;
- every corpus is standardised with its own feature statistics; the statistics of the challenge training set are frozen
  into the model and used at test time;
- AdamW (lr 3·10⁻⁴, weight decay 0.05), one-cycle schedule, 2,000 steps;
- optional last stage: 200 steps at lr 10⁻⁵ on the challenge identities only (48 identities × 4 samples per batch);
- 10 networks with different seeds per model; their cosine scores are averaged.

| Recipe | Temperature | Hidden dropout p | Last stage |
|---|:---:|:---:|:---:|
| `tau003_p05_ft200` | 0.03 | 0.5 | ✓ |
| `tau002_ft200` | 0.02 | 0.3 | ✓ |
| `tau002` | 0.02 | 0.3 | – |

**Training data:** 7,392 identities (70 challenge + 1,211 VoxCeleb1 + 6,111 VoxCeleb2), 157,277 face images,
86,968 voice clips.

### 3. Ensemble

Three model generations, each trained on three voice-input sets:

| Generation | Face input | Voice input | Model tags |
|---|---|---|---|
| 1 | CLIP | set c, e or f | `e3q_<set>_<recipe>` |
| 2 | CLIP + PE-Core | set c, e or f + ResNet293 | `e3rcpel_<set>_<recipe>` |
| 3 | CLIP | set c, e or f + ResNet293 | `e3r_<set>W_<recipe>` |

Voice sets: **c** = WavLM + age/gender + ECAPA + ReDimNet-B6; **e** = ECAPA + ReDimNet-B6 + ReDimNet-M;
**f** = WavLM + age/gender + ECAPA + ReDimNet-B6 + ReDimNet-M.

Each condition uses one recipe, chosen by a fixed rule on training-side validation (`configs/cell_policy.md`):

| Condition | Recipe |
|---|---|
| No-gender English | `tau003_p05_ft200` |
| No-gender Bangla | `tau002_ft200` |
| Gender English | `tau003_p05_ft200` |
| Gender Bangla | `tau002` |

**Gender experts** (same-gender conditions only). Per seed, one network is trained on male and one on female identities.
For each trial the two scores are mixed with p(gender | voice) from a logistic regression fitted on training voices.
Four expert members are added: set c with CLIP faces, and sets c, e, f + ResNet293 with CLIP + PE-Core faces.

In total, the final system has 27 base models (270 networks) and 4 expert members (80 networks).

### 4. Anchor consistency term

The anchor consistency term (`src/anchor.py`) checks whether the face and the voice of a trial agree with the same pair of
averaged embeddings from public corpora. The anchor set covers 7,405 identities of VoxCeleb1, VoxCeleb2 and MAV-Celeb v2;
each anchor is stored under the corpus's anonymous ID and consists of a face anchor F_i (mean ArcFace embedding) and a voice
anchor V_i (mean ReDimNet-B6 embedding). For face *f* and voice *v*, let *X* be the anchor whose face anchor is closest to
*f* and *Y* the anchor whose voice anchor is closest to *v*:

```
m = σ((cos(f, F_X) − 0.50) / 0.05) · (cos(v, V_X) − 0.45)  +  σ((cos(v, V_Y) − 0.65) / 0.05) · (cos(f, F_Y) − 0.30)
```

If the face is close to a face anchor, the voice is checked against that pair's voice anchor (bonus or penalty), and vice
versa. Otherwise both gates stay closed and *m* ≈ 0. The term is not trained and only adds a number to the trial's score;
it never outputs or reports an identity. Its constants were set on the MAV-Celeb v2 validation set; the voice gate was also
checked for false openings on the challenge training identities without an anchor (faces 0.00 %, voices 0.08 %).

### 5. Final score

Per condition: **final score = mean of the members' scores + w · m**. The weight w is fixed by one rule on the DEV2-short
validation set: w = 3 · std(base score of the condition's models) / std(base score of a reference model family with a
learned temperature). This keeps the balance between base models and anchor term constant when a sharper temperature
narrows the base-score range. Members and weights: `configs/final_system.json`.

In `configs/final_system.json` the anchor term (`anchor`) appears as one more member of a weighted mean, with weight W = n · w for n
members (for example 14.453 = 9 × 1.606 in the no-gender English condition). The weighted mean
(Σ sᵢ + W · m) / (n + W) equals n / (n + W) · (mean of sᵢ + w · m): a positive constant per condition times the final score
above, so the ranking of trials and the EER are the same.

### 6. Validation (training-side only)

- **Cross-validation:** 7 folds by identity over the 70 challenge training identities, with any-gender and same-gender
  negatives.
- **DEV2:** MAV-Celeb v2 (83 identities, English and Hindi) as a four-condition validation set; **DEV2-short** uses the
  same trials with voices cut to the clip-length distribution of the challenge training set.

---

## Compliance with the challenge rules

- **Per-trial scoring.** Every score depends only on the trial's own face and voice. There is no clustering or other
  processing across trials. Scoring a trial alone, in a batch or in reversed order gives the same score up to float
  rounding (≤ 3·10⁻⁷).
- **No use of progress or evaluation data.** Evaluation files are only scored. All statistics, weights, thresholds and
  model choices come from the challenge training set, the external corpora and the MAV-Celeb v2 validation set.
- **No score normalisation** of any kind.
- **Frozen public encoders.** No encoder is trained, fine-tuned or adapted by us. No Bengali data was added or selected
  for any part of the system (training, adaptation, calibration, thresholds or weights). The public encoders are used as
  released; they were trained by their authors on general corpora that are not filtered by language (e.g. VoxCeleb,
  VoxBlink2, CN-Celeb for the speaker encoders).
- **Ensembles** use fixed weights from training-side validation.
- **External data** is public and listed file by file in `data/manifests/`.
- **Overlap with public data.** The anchor consistency term helps when a test identity also appears in the public
  corpora. The anchor set and all constants were fixed without any progress or evaluation data, and no overlap with test
  identities was ever measured on test data. Names are never used; corpus metadata (gender labels) is used only to train
  the base models and gender experts.

---

## Repository structure

```
FLAG2027/
├── README.md
├── LICENSE                        # MIT
├── requirements.txt               # training, inference, anchor term, combination
├── requirements-extraction.txt    # feature extraction
├── requirements-data.txt          # download of the external corpora
├── requirements/                  # pip freeze of the environments that were used
├── scripts/
│   └── download_models.py         # downloads all pretrained encoders
├── configs/
│   ├── recipes.json               # training recipes (+ their validation results)
│   ├── cell_policy.json / .md     # recipe choice per condition
│   └── final_system.json          # members and weights per condition (the submission)
├── models/                        # <tag>.scores.json per member; checkpoints from the GitHub release
├── precomputed/                   # DEV2-short scores used by the anchor-weight rule
├── data/
│   └── manifests/                 # exact file lists, genders and overlap lists of the external corpora
└── src/
    ├── common.py                  # paths, trial lists, recipes, EER
    ├── model.py                   # data sources, two-tower model, training, cross-validation, scoring
    ├── anchor.py                  # anchor consistency term
    ├── dev2.py                    # MAV-Celeb v2 validation set
    ├── extract_features.py        # CLIP, ArcFace, WavLM, wav2vec2 age/gender, ECAPA, ReDimNet-B6/M
    ├── extract_pecore.py          # PE-Core
    ├── extract_resnet293.py       # WeSpeaker ResNet293
    ├── train_base.py              # one base model (10 seeds)
    ├── train_family.py            # one recipe on several voice-input sets
    ├── train_gender_experts.py    # gender-expert member
    ├── score_dev2short.py         # DEV2-short scores of trained models
    ├── combine.py                 # submission from stored member scores
    ├── infer.py                   # submission from checkpoints
    ├── data_prep/                 # download and sampling of the external corpora
    └── validation/                # experiments behind the design choices
```

---

## Installation

Python 3.12, an NVIDIA GPU with CUDA 12.1.

```bash
git clone https://github.com/prathamg007/FLAG2027.git
cd FLAG2027
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121              # training, inference
pip install -r requirements-extraction.txt --extra-index-url https://download.pytorch.org/whl/cu121   # feature extraction
pip install -r requirements-data.txt                                                                 # external data download
python scripts/download_models.py                                                                    # pretrained encoders
```

Training, inference and combination need only `torch`, `numpy`, `scikit-learn` and `scipy`. Separate virtual environments
for the three files avoid version conflicts. The environments actually used are recorded in `requirements/`:

| File | Environment |
|---|---|
| `freeze-training.txt` | training of model generations 2 and 3 and of the gender experts |
| `freeze-extraction.txt` | PE-Core and WeSpeaker ResNet293 feature extraction |
| `freeze-workstation.txt` | data download; CLIP, ArcFace, WavLM, wav2vec2 age/gender, ECAPA and ReDimNet extraction; training of generation 1; all verification runs of this release |

**Checkpoints.** The 31 trained members (4.6 GB) are attached to the GitHub release `v1.0` of this repository (or download them from the Releases page):

```bash
gh release download v1.0 --pattern "*.ckpt.pt" --dir models
```

---

## Data

No media or features are distributed here. Paths are set with environment variables:

| Variable | Contents | Default |
|---|---|---|
| `FLAG_DATA` | challenge data as released: `train_set/` and `eval_set/evaluation_phase/` | `data/` |
| `FLAG_EXT` | external media: `vox1_faces/`, `vox1_audio/`, `vox2/`, `mavold/` | `$FLAG_DATA/ext_data` |
| `FLAG_FEAT` | feature tables written by the extraction scripts | `features/` |
| `FLAG_MODELS` | checkpoints and score files | `models/` |

| Corpus | Identities | Role | Source |
|---|---|---|---|
| MAV-Celeb v4 training set | 70 | training | challenge release |
| VoxCeleb1 | 1,211 | training, anchor set | faces: VGG cross-modal biometrics release `zippedFaces.tar.gz`; audio: Hugging Face `humanify/voxceleb1_dev` |
| VoxCeleb2 (sampled) | 6,111 | training, anchor set | Hugging Face `blueskyheaven/voxceleb2-mp4-binary`: one frame and up to 10 s of audio per clip |
| MAV-Celeb v2 | 83 | validation, anchor set | organisers' public release |

The external media are obtained with the scripts in `src/data_prep/`. Copy `data/manifests/vox1_meta.csv` and
`data/manifests/vox2_meta.csv` into `$FLAG_EXT` first (the build scripts read the identity metadata from there):

```bash
python src/data_prep/build_vox1.py faces            # VoxCeleb1 face crops (needs zippedFaces.tar.gz in $FLAG_EXT)
python src/data_prep/download_vox1_audio.py         # VoxCeleb1 audio
python src/data_prep/fetch_vox2.py index            # VoxCeleb2: index of the parquet shards
python src/data_prep/fetch_vox2.py fetch 10         # VoxCeleb2: sampled frames and audio
python src/data_prep/download_mavceleb_v2.py        # MAV-Celeb v2 archive
python src/data_prep/build_mavceleb_v2.py mavceleb_download/mavceleb_v2_complete.zip v2
```

The models were trained on the files listed in `data/manifests/`; the code reads these lists. Every file is identified by
a fixed key path (for example `D:\FLAG27\ext_data\vox2\id00012\33_113_078.jpg`), which `src/common.py` maps to
`$FLAG_EXT/vox2/id00012/33_113_078.jpg` (or `$FLAG_DATA/...` for challenge files) on any operating system.

The `data_prep` scripts only fetch the media into this layout; the lists they write into `$FLAG_EXT` (with plain disk
paths) are for inspection only. Extraction, training and inference always use the shipped lists in `data/manifests/`.
`data/manifests/v4_train_durations.npy` holds the clip lengths of the challenge training voices (no audio); it defines
the clip-length distribution of the DEV2-short validation set.

---

## Reproducing the results

There are three levels, from seconds to a full retraining.

### Level 1: rebuild the submission from stored scores (CPU, seconds)

Needs the evaluation trial lists in `$FLAG_DATA/eval_set/` and the packages of `requirements.txt` (`torch` is imported;
a CPU build is enough, no GPU and no features are needed).

```bash
python src/combine.py final --spec configs/final_system.json
```

The four score files in `models/final.zip` are identical to those of the submitted archive (only the zip timestamps differ).

### Level 2: inference from the released checkpoints

Needs the checkpoints in `models/` and the evaluation features (step 1 below, datasets `eval` and, for the anchor term,
`vox1`, `vox2`, `mavold` with `arc`, `ecapa` and `redim`).

```bash
python src/infer.py --check
```

This recomputes every member on the evaluation trials, prints the largest difference to the stored scores and writes
`models/final.zip`.

### Level 3: full training from scratch

**Step 1: features**

```bash
for ds in mav eval vox1 vox2 mavold; do
  python src/extract_features.py $ds clip,arc,wavlm,w2vag,ecapa,redim,redimm
done
python src/extract_features.py mavolds wavlm,w2vag,ecapa,redim,redimm        # DEV2-short voices
python src/extract_pecore.py mav,eval,mavold,vox1,vox2
python src/extract_resnet293.py mav,eval,mavold,mavolds,vox1,vox2
```

**Step 2: base models**, 3 recipes × 3 generations × 3 voice sets, each scored on DEV2-short:

```bash
for r in tau003_p05_ft200 tau002_ft200 tau002; do
  FV_VOX1_M16=1 python src/train_family.py --prefix e3q_ --recipe $r --faces clip \
    --sets "c=wavlm,w2vag,ecapa,redim;e=ecapa,redim,redimm;f=wavlm,w2vag,ecapa,redim,redimm"
  python src/train_family.py --prefix e3rcpel_ --recipe $r --faces clip,pel \
    --sets "c=wavlm,w2vag,ecapa,redim,wsr293;e=ecapa,redim,redimm,wsr293;f=wavlm,w2vag,ecapa,redim,redimm,wsr293"
  python src/train_family.py --prefix e3r_ --recipe $r --faces clip \
    --sets "cW=wavlm,w2vag,ecapa,redim,wsr293;eW=ecapa,redim,redimm,wsr293;fW=wavlm,w2vag,ecapa,redim,redimm,wsr293"
done
```

Reference family for the anchor-weight rule (learned temperature; its DEV2-short scores are already in `precomputed/`):

```bash
export FV_VOX1_M16=1
python src/train_base.py e3c_light_b6   --voices wavlm,w2vag,ecapa,redim       --recipe base
python src/train_base.py e3e_light_lean --voices ecapa,redim,redimm            --recipe base
python src/train_base.py e3f_light_b6m  --voices wavlm,w2vag,ecapa,redim,redimm --recipe base
python src/score_dev2short.py e3c_light_b6 e3e_light_lean e3f_light_b6m
```

**Step 3: gender experts**

```bash
python src/train_gender_experts.py e3rgx_c_tau003_p05_ft200     --faces clip     --voices wavlm,w2vag,ecapa,redim
python src/train_gender_experts.py e3rcpelgx_c_tau003_p05_ft200 --faces clip,pel --voices wavlm,w2vag,ecapa,redim,wsr293
python src/train_gender_experts.py e3rcpelgx_e_tau003_p05_ft200 --faces clip,pel --voices ecapa,redim,redimm,wsr293
python src/train_gender_experts.py e3rcpelgx_f_tau003_p05_ft200 --faces clip,pel --voices wavlm,w2vag,ecapa,redim,redimm,wsr293
```

**Step 4: anchor consistency term**

```bash
python src/anchor.py anchor
```

**Step 5: submission**

```bash
python src/combine.py final --spec configs/final_system.json
# or recompute the anchor-term weights with the rule:
python src/combine.py final --families "ngE=<tags>|ngB=<tags>|gE=<tags>|gB=<tags>"
```

**Notes on exact reproduction**

- Training is deterministic per seed on a given GPU; a base model retrained with this code matched the released one
  exactly.
- `FV_VOX2_OLD=1` (default in `src/common.py`) selects the VoxCeleb2 file list the models were trained on
  (`data/manifests/vox2_manifest_pass1.tsv`).
- `FV_VOX1_M16=1` (generation 1 and the reference family) limits VoxCeleb1 voices to the first 16 per identity when
  ReDimNet-M is an input, as in the original runs.

---

## Validation tools

| Script | Purpose |
|---|---|
| `src/validation/recipe_study.py` | measures recipes on DEV2, DEV2-short and cross-validation (`configs/recipes.json`) |
| `src/validation/cell_policy.py` | chooses the recipe per condition with a fixed rule (`configs/cell_policy.*`) |
| `src/validation/cv_dump.py`, `ensemble_cv.py` | per-trial cross-validation scores and paired comparison of ensembles (e.g. the gender experts) |
| `src/validation/dev2short_system.py` | DEV2-short result of a complete system with the anchor term |
| `src/validation/anchor_checks.py` | challenge training identities with an anchor; false gate openings on the others |

---

## Pretrained models and licences

| Model | Source | Licence |
|---|---|---|
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | MIT |
| PE-Core-L/14-336 | `timm/vit_pe_core_large_patch14_336.fb` | Apache 2.0 |
| WavLM-base-plus | `microsoft/wavlm-base-plus` | see model card |
| wav2vec2 age/gender | `audeering/wav2vec2-large-robust-24-ft-age-gender` | CC BY-NC-SA 4.0 |
| ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` | Apache 2.0 |
| ReDimNet-B6 / M | `IDRnD/ReDimNet` (torch.hub) | MIT |
| WeSpeaker ResNet293-LM | `Wespeaker/wespeaker-voxceleb-resnet293-LM` | see model card |
| ArcFace (buffalo_l) | InsightFace | non-commercial research use |

The datasets are subject to their own terms; this repository contains only file lists, not media.

## Licence

The code in this repository is released under the [MIT licence](LICENSE). Pretrained encoders and datasets keep their
own licences (see the table above).
