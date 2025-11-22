import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Path absoluto al proyecto
project_dir = r'C:\Users\CHRISTIAN\Documents\EvaluacionEjerciciosIA'
sys.path.append(os.path.join(project_dir, 'src'))

from utils import calculate_angle  # Importa la función de src/utils.py

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Evaluación de Ejercicios con Videos Pregrabados")

# Carga YOLOv8s-pose
model_pose = YOLO(os.path.join(project_dir, 'models/yolov8s-pose.pt'))

# Define el modelo LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, num_classes=6):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Carga el modelo entrenado
model = LSTMModel(input_size=30, hidden_size=64, num_layers=2, num_classes=6).to(device)
model.load_state_dict(torch.load(os.path.join(project_dir, 'models/lstm_model.pth')))
model.eval()

# LabelEncoder
le = LabelEncoder()
le.classes_ = np.array(['pushups_down', 'pushups_up', 'situp_down', 'situp_up', 'squats_down', 'squats_up'])

# Buffer para secuencias
buffer = []
seq_length = 10

# Subir video
uploaded_file = st.file_uploader("Carga un video de ejercicio", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Guarda el video temporalmente
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture("temp_video.mp4")
    placeholder = st.empty()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total de frames: {frame_count}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Fin del video.")
            break

        # Procesa el frame con YOLOv8s-pose
        results = model_pose(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy()
        if len(keypoints) > 0:
            keypoints = keypoints[0]  # Primera persona
            keypoints = keypoints / np.array([640, 480])  # Normaliza (ajusta si el video tiene otra resolución)

            # Calcula ángulos (índices COCO)
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            right_shoulder = keypoints[6]
            right_elbow = keypoints[8]
            right_wrist = keypoints[10]
            left_hip = keypoints[11]
            left_knee = keypoints[13]
            left_ankle = keypoints[15]
            right_hip = keypoints[12]
            right_knee = keypoints[14]
            right_ankle = keypoints[16]

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Forma el frame_data: 26 keypoints clave + 4 ángulos
            keypoint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # Nariz a caderas
            selected_keypoints = keypoints[keypoint_indices].flatten()
            frame_data = np.concatenate((selected_keypoints, [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]))

            # Añade al buffer
            buffer.append(frame_data)
            if len(buffer) > seq_length:
                buffer.pop(0)

            # Predice si el buffer está lleno
            if len(buffer) == seq_length:
                sequence = np.array(buffer)
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(sequence)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    predicted_label = le.inverse_transform([predicted.cpu().numpy()])[0]
                    confidence = probabilities[0][predicted.cpu().numpy()].item() * 100

                # Muestra en Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                with placeholder.container():
                    st.image(img_pil, caption=f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{frame_count}", use_column_width=True)
                    st.write(f"Ejercicio predicho: {predicted_label} (Confianza: {confidence:.1f}%)")
                    if confidence < 80:
                        st.warning("Alerta: Baja confianza - Verifica la postura o calidad del video.")
                    if 'down' in predicted_label.lower():
                        st.info("Fase descendente detectada.")
                    elif 'up' in predicted_label.lower():
                        st.success("Fase ascendente detectada.")

        cv2.waitKey(1)

    cap.release()
    os.remove("temp_video.mp4")  # Limpia el archivo temporal