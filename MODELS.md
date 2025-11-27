# Model Files

## Download Pre-trained Models

The trained model files are too large for GitHub. Download them here:

### Option 1: Google Drive (Recommended)

[Download Models from Google Drive](YOUR_GOOGLE_DRIVE_LINK)

The zip file contains:

- `best_efficientnet_focal.pth` (183 MB)
- `best_vit_model.pth` (345 MB)
- `class_mapping.json` (1 KB)

### Option 2: Hugging Face

Coming soon!

## Installation

1. Download the models
2. Extract to `models/` folder:

```
GastroVision_Docker/
└── models/
    ├── best_efficientnet_focal.pth
    ├── best_vit_model.pth
    └── class_mapping.json
```

3. Verify files:

```bash
ls -lh models/
```

## Model Details

| File                          | Size    | Description                          |
| ----------------------------- | ------- | ------------------------------------ |
| `best_efficientnet_focal.pth` | ~183 MB | EfficientNet-B3 (94.27% accuracy)    |
| `best_vit_model.pth`          | ~345 MB | Vision Transformer (95.50% accuracy) |
| `class_mapping.json`          | 1 KB    | Class names mapping                  |

**Total:** ~528 MB
