# OCR-based Data Logger

This project is an **Optical Character Recognition (OCR)** system designed as a *data logger*. It uses **YOLOv8** for text area detection and a **CRNN with CTC** for character recognition.

## Dataset

- **YOLO Dataset (for text area detection):** https://app.roboflow.com/plothole-detection/sample-z1q81/5(#)
- **OCR Dataset (for text recognition):** https://drive.google.com/drive/folders/1l0STrSRjec8TFqYEmueUiAa9eMvLIz0O?usp=sharing(#)

## File Descriptions

| File/Folder                   | Description                                                                                 |
|-------------------------------|---------------------------------------------------------------------------------------------|
| `Object Detection/display.pt` | YOLOv8 custom model weights for text area detection.                                        |
| `OCR/crnn_ocr.h5`             | CRNN model weights for text recognition.                                                    |
| `integration.py`              | Script to run inference for both detection and OCR on input images.                         |
| `custom-training.ipynb`       | Script to train the YOLO model with a custom dataset.                                       |
| `train-v2.ipynb`              | Script to train the OCR CRNN model.                                                         | 
| `textgenerate.ipynb`          | Script to generate synthetic data.                                                          |
| `README.md`                   | This documentation file.                                                                    |
