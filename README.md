# AI-Powered Appliance Damage Assessment Platform

A comprehensive AI system for detecting and assessing damage to consumer electronics and appliances, with integrated fraud detection capabilities.

## Project Structure

```
AI-Appliance-Assessment/
├── configs/
│   └── config.py              # All project configuration
├── datasets/                   # Dataset storage
│   ├── appliance_detector/    # Appliance detection dataset
│   └── damage_detector/        # Damage detection datasets
│       └── phone/             # Your phone crack dataset
├── models/
│   ├── appliance_detector/    # Trained appliance models
│   └── damage_detector/       # Trained damage models
├── fraud_detection/           # Fraud detection engine
├── risk_engine/               # Risk scoring engine
├── report_engine/             # Report generation
├── dashboard/                 # Streamlit dashboard
├── scripts/                   # Training and inference scripts
│   ├── train_appliance_detector.py
│   ├── train_damage_detector.py
│   ├── inference.py
│   └── evaluate.py
├── utils/                     # Utility functions
├── tests/                     # Unit tests
├── main.py                    # Main entry point
└── requirements.txt           # Python dependencies
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd /Users/sneh/Downloads/AI-Appliance-Assessment
pip install -r requirements.txt
```

### 2. Your Existing Dataset

Your phone crack dataset is located at:
```
/Users/sneh/Downloads/cracked screen.v1i.yolov8 2/
```

To use it with this project, you can either:

**Option A: Move/Copy manually**
```bash
cp -r "/Users/sneh/Downloads/cracked screen.v1i.yolov8 2"/* \
  "/Users/sneh/Downloads/AI-Appliance-Assessment/datasets/damage_detector/phone/"
```

**Option B: Create symlink**
```bash
ln -s "/Users/sneh/Downloads/cracked screen.v1i.yolov8 2" \
  "/Users/sneh/Downloads/AI-Appliance-Assessment/datasets/damage_detector/phone"
```

### 3. Training Your Phone Crack Detector

Since your dataset has the class "cracked" and the project expects "screen_crack", I've configured the training script to handle this mapping. Run:

```bash
python scripts/train_damage_detector.py \
  --appliance phone_crack \
  --dataset /Users/sneh/Downloads/cracked\ screen.v1i.yolov8\ 2 \
  --epochs 50 \
  --batch-size 8 \
  --device cuda
```

### 4. Inference

```bash
python scripts/inference.py \
  --image /path/to/test/image.jpg \
  --save-vis \
  --output-dir output
```

## Supported Appliances

| Appliance | Damage Types |
|-----------|-------------|
| Phone | screen_crack, display_lines, camera_crack, body_damage |
| Television | screen_crack, dead_pixels, display_lines, panel_damage |
| Laptop | screen_crack, keyboard_damage, hinge_damage, body_dent |
| Tablet | screen_crack, glass_shatter, body_damage |
| Monitor | screen_crack, dead_pixels, display_lines |
| Refrigerator | dent, rust, door_seal_damage, surface_scratch |
| Washing Machine | drum_damage, door_damage, panel_dent, rust |
| Air Conditioner | fin_damage, body_dent, rust |
| Microwave | door_damage, body_dent, rust, glass_damage |

## Phases Completed

- ✅ Phase 1: Project structure and setup
- ⏳ Phase 2: Appliance detector training pipeline
- ⏳ Phase 3: Damage detector training pipeline  
- ✅ Phase 4: Fraud detection engine
- ✅ Phase 5: Risk engine
- ⏳ Phase 6: Dashboard
- ⏳ Phase 7: API deployment
