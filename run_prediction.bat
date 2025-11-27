@echo off
REM GastroVision Prediction Script by Nishant Joshi

if "%1"=="" (
    echo Usage: run_prediction.bat path_to_image
    echo Example: run_prediction.bat test_images/Colon_polyps/image.jpg
    exit /b 1
)

docker run --rm -v "%cd%/test_images:/app/test_images" gastrovision:latest python src/predict.py --image %1 --models_dir models/
