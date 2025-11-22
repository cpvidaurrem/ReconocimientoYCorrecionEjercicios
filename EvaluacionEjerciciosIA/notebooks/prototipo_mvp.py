import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Path absoluto al proyecto (ajusta si tu ruta es diferente)
project_dir = r'C:\Users\CHRISTIAN\Documents\EvaluacionEjerciciosIA'
sys.path.append(os.path.join(project_dir, 'src'))

from utils import calculate_angle  # Importa después del append

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Prototipo Integrado - Evaluación de Ejercicios con LSTM")

# Carga YOLOv8s-pose (versión 8.3.2 estable con CUDA 12.6)
model_pose = YOLO(os.path.join(project_dir, 'models/yolov8s-pose.pt'))

# Define el modelo LSTM (misma arquitectura que en Fase 3)
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

# Carga el modelo entrenado (versión estable con PyTorch 2.8.0+cu126)
model = LSTMModel(input_size=30, hidden_size=64, num_layers=2, num_classes=6).to(device)
model.load_state_dict(torch.load(os.path.join(project_dir, 'models/lstm_model.pth')))
model.eval()

# LabelEncoder (basado en las clases conocidas del entrenamiento)
le = LabelEncoder()
le.classes_ = np.array(['pushups_down', 'pushups_up', 'situp_down', 'situp_up', 'squats_down', 'squats_up'])

# Buffer para secuencias (seq_length=10)
buffer = []
seq_length = 10

# Captura video
cap = cv2.VideoCapture(0)
placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("No se puede acceder a la webcam.")
        break

    # Procesa el frame con YOLOv8s-pose
    results = model_pose(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy()  # 17 keypoints por persona (1x17x2)
    if len(keypoints) > 0:
        keypoints = keypoints[0]  # Primera persona (17x2)
        # Normaliza keypoints (asume frame 640x480, como en preprocesamiento)
        keypoints = keypoints / np.array([640, 480])

        # Calcula ángulos (índices COCO: left_shoulder=5, right_shoulder=6, etc.)
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

        # Forma el frame_data: 26 valores de keypoints clave + 4 ángulos = 30
        keypoint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # Nariz a caderas (13x2=26)
        selected_keypoints = keypoints[keypoint_indices].flatten()
        frame_data = np.concatenate((selected_keypoints, [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]))

        # Añade al buffer
        buffer.append(frame_data)
        if len(buffer) > seq_length:
            buffer.pop(0)

        # Predice si el buffer está lleno
        if len(buffer) == seq_length:
            sequence = np.array(buffer)  # (10, 30)
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, 10, 30)
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
                st.image(img_pil, caption="Video en tiempo real", use_column_width=True)
                st.write(f"Ejercicio predicho: {predicted_label} (Confianza: {confidence:.1f}%)")
                if confidence < 80:
                    st.warning("Alerta: Baja confianza - Verifica tu postura o iluminación.")
                if 'down' in predicted_label.lower():
                    st.info("Fase descendente: Baja más profundo para mejor profundidad.")
                elif 'up' in predicted_label.lower():
                    st.success("Fase ascendente: Buena fluidez!")

    cv2.waitKey(1)  # ~30 fps

cap.release()