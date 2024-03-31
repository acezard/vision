# Vision-Language playground

This is a simple playground to test vision-language models with Gradio and LM Studio.

Demo feature:
- Accept a directory of images as input
- For each image, the vision model will generate a caption
- For each caption, the client-side app will categorize the image based on the caption:
  - Photograph
  - Document
  - Uncategorized
- The results will be displayed to the user as text and bar chart
- Includes client-side logging

# Requirements
- Python 3.10+
- LM Studio
- Approx. 16GB of VRAM

# How to run

## Client
In watch mode :
```bash
gradio vision.py
```

In server mode :
```bash
python vision.py
```

## Server
Use `model_config.json` in **LM Studio** in server mode with the following models :
- llava-v1.5-13b-Q5_K_M.gguf
- mmproj-model-f16.gguf
