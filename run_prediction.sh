#!/bin/bash
# GastroVision Prediction Script by Nishant Joshi

if [ -z "$1" ]; then
    echo "Usage: ./run_prediction.sh <path_to_image>"
    echo "Example: ./run_prediction.sh test_images/Colon_polyps/image.jpg"
    exit 1
fi

docker run --rm \
    -v "$(pwd)/test_images:/app/test_images" \
    gastrovision:latest \
    python src/predict.py --image "$1" --models_dir models/
