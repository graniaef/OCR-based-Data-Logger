import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

def preprocess_image(image, resize_width=200, resize_height=100):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (resize_width, resize_height))
    return np.expand_dims(resized_image, axis=-1)

def decode_predictions_ctc(y_pred):
    input_length = np.full((y_pred.shape[0],), y_pred.shape[1], dtype=np.int32)
    decoded_predictions, _ = tf.keras.backend.ctc_decode(y_pred, input_length)

    decoded_texts = []
    for pred in decoded_predictions[0].numpy():
        decoded_text = ''.join([str(char) if char != 10 else '.' for char in pred if char != -1])
        decoded_texts.append(decoded_text)

    return decoded_texts

def detect_and_predict(yolo_model_path, h5_model_path, output_folder, excel_output_path, video_output_path):
    yolo_model = YOLO(yolo_model_path)

    h5_model = load_model(h5_model_path, compile=False)

    video = cv2.VideoCapture(1)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    prediction_data = []

    os.makedirs(output_folder, exist_ok=True)

    start_time = time.time() 

    while True:
        ret, frame = video.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        timestamp = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        results = yolo_model.predict(source=frame, save=False, conf=0.5)

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)

            cropped_image = frame[y1:y2, x1:x2]

            preprocessed_image = preprocess_image(cropped_image)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

            y_pred = h5_model.predict(preprocessed_image)
            decoded_texts = decode_predictions_ctc(y_pred)

            cropped_filename = f"{output_folder}/frame_{int(elapsed_time)}_object_{i}.jpg"
            cv2.imwrite(cropped_filename, cropped_image)

            print(f"  📌 {timestamp} - Object {i}: {decoded_texts[0]}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, decoded_texts[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)

            prediction_data.append({
                "Timestamp": timestamp,
                "Prediction": decoded_texts[0]
            })

        out.write(frame)

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(prediction_data)

    df.to_excel(excel_output_path, index=False)
    print(f"\n✅ Processing complete! Predictions saved to {excel_output_path}")

    print("\n📊 Preview Hasil Prediksi:")
    print(df.head())

yolo_model_path = "displayv5.pt"
h5_model_path = "modelV2.h5"
output_folder = "demo"
excel_output_path = "demo.xlsx"
video_output_path = "demo.mp4"


detect_and_predict(yolo_model_path, h5_model_path, output_folder, excel_output_path, video_output_path)
