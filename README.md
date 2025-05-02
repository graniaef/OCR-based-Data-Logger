# OCR-based Data Logger

This project is an **Optical Character Recognition (OCR)** system designed as a *data logger*. It uses **YOLOv8** for text area detection and a **CRNN with CTC** for character recognition.

## Dataset

- **YOLO Dataset (for text area detection):** [Link X](#)
- **OCR Dataset (for text recognition):** [Link Y](#)

## File Descriptions

| File/Folder                   | Description                                                                                 |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `Object Detection/display.pt` | YOLOv8 custom model weights for text area detection.                                        |
| `OCR/crnn_ocr.h5`             | CRNN model weights for text recognition.                                                   |
| `integration.py`              | Script to run inference for both detection and OCR on input images.                        |
| `custom-training.ipynb`       | Script to train the YOLO model with a custom dataset.                                      |
| `train-v2.ipynb`              | Script to train the OCR CRNN model.                                                        |
| `requirements.txt`            | List of required dependencies.                                                             |
| `README.md`                   | This documentation file.                                                                   |
