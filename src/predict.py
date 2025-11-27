#!/usr/bin/env python3
"""
GastroVision Inference Script
Predicts gastrointestinal conditions from endoscopy images by Nishant Joshi
"""

import argparse
import json
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class EnsembleModel(nn.Module):
    # EfficeinetNetB3 + ViT
    def __init__(self, model1, model2, weight1=0.4, weight2=0.6):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return self.weight1 * out1 + self.weight2 * out2


def load_models(efficientnet_path, vit_path, class_mapping_path, device):
    # Load trained ensemble models

    # Load class mapping
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)

    num_classes = len(class_mapping["class_names"])

    # Create EfficientNet-B3
    efficientnet = timm.create_model("efficientnet_b3", pretrained=False)
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(efficientnet.classifier.in_features, num_classes)
    )

    # Load weights
    checkpoint = torch.load(efficientnet_path, map_location=device)
    efficientnet.load_state_dict(checkpoint["model_state_dict"])
    efficientnet = efficientnet.to(device)
    efficientnet.eval()

    # Create Vision Transformer
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    vit.head = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(vit.head.in_features, num_classes)
    )

    # Load weights
    checkpoint = torch.load(vit_path, map_location=device)
    vit.load_state_dict(checkpoint["model_state_dict"])
    vit = vit.to(device)
    vit.eval()

    # Create ensemble
    ensemble = EnsembleModel(efficientnet, vit, weight1=0.4, weight2=0.6)
    ensemble = ensemble.to(device)
    ensemble.eval()

    return ensemble, class_mapping


def preprocess_image(image_path):
    # Preprocess image for model input

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(model, image_tensor, class_mapping, device):
    # Make prediction

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = probabilities.max(1)

    predicted_label = class_mapping["class_names"][predicted_class.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="GastroVision Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model paths
    models_dir = Path(args.models_dir)
    efficientnet_path = models_dir / "best_efficientnet_focal.pth"
    vit_path = models_dir / "best_vit_model.pth"
    class_mapping_path = models_dir / "class_mapping.json"

    # Check files exist
    if not efficientnet_path.exists():
        print(f"❌ EfficientNet model not found: {efficientnet_path}")
        sys.exit(1)
    if not vit_path.exists():
        print(f"❌ ViT model not found: {vit_path}")
        sys.exit(1)
    if not class_mapping_path.exists():
        print(f"❌ Class mapping not found: {class_mapping_path}")
        sys.exit(1)

    # Load models
    print("Loading models...")
    model, class_mapping = load_models(
        efficientnet_path, vit_path, class_mapping_path, device
    )
    print("Models loaded!")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image_tensor = preprocess_image(args.image)

    # Make prediction
    print("Making prediction...")
    predicted_label, confidence, all_probs = predict(
        model, image_tensor, class_mapping, device
    )

    # Display results
    print("\n" + "=" * 60)
    print("GASTROVISION PREDICTION RESULTS")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nAll Class Probabilities:")
    for i, class_name in enumerate(class_mapping["class_names"]):
        print(f"   {class_name}: {all_probs[i]*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
