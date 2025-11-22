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

from utils import calculate_angle

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Evaluación de Ejercicios con Videos Pregrabados - Visualización Completa")

# Carga YOLOv8s-pose
@st.cache_resource
def load_yolo_model():
    return YOLO(os.path.join(project_dir, 'models/yolov8s-pose.pt'))

model_pose = load_yolo_model()

# Define el modelo LSTM (debe coincidir con el del entrenamiento)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # IMPORTANTE: Mismo dropout que en entrenamiento
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, 
                                   dropout=dropout if num_layers > 1 else 0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Carga el modelo entrenado (CORREGIDO)
@st.cache_resource
def load_lstm_model():
    model = LSTMModel(input_size=30, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3).to(device)
    
    # Intenta cargar el checkpoint completo primero
    checkpoint_path = os.path.join(project_dir, 'models/best_lstm_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Verificar si es un checkpoint completo o solo state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Es un checkpoint completo (nuevo formato)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"✅ Modelo cargado (Época {checkpoint.get('epoch', 'N/A')}, "
                      f"Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")
        else:
            # Es solo state_dict (formato antiguo)
            model.load_state_dict(checkpoint)
            st.success("✅ Modelo cargado (formato antiguo)")
    else:
        st.error(f"❌ No se encontró el modelo en: {checkpoint_path}")
        st.stop()
    
    model.eval()
    return model

model = load_lstm_model()

# LabelEncoder (debe coincidir con el entrenamiento)
le = LabelEncoder()
le.classes_ = np.array(['pushups_down', 'pushups_up', 'situp_down', 'situp_up', 'squats_down', 'squats_up'])

# Configuración de visualización
st.sidebar.header("⚙️ Configuración")
confidence_threshold = st.sidebar.slider("Umbral de confianza (%)", 0, 100, 70, 5)
show_angles = st.sidebar.checkbox("Mostrar ángulos", value=True)
show_confidence = st.sidebar.checkbox("Mostrar confianza", value=True)

# Buffer para secuencias
buffer = []
seq_length = 10

# Información
st.sidebar.info(f"""
**ℹ️ Información del Sistema:**
- Dispositivo: {device}
- Longitud de secuencia: {seq_length} frames
- Clases: {len(le.classes_)}
""")

# Subir video
uploaded_file = st.file_uploader("📹 Carga un video de ejercicio", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Guarda el video temporalmente
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture(temp_video_path)
    
    if not cap.isOpened():
        st.error("❌ No se pudo abrir el video")
        os.remove(temp_video_path)
        st.stop()
    
    # Información del video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Frames", frame_count)
    col2.metric("FPS", fps)
    col3.metric("Resolución", f"{width}x{height}")
    col4.metric("Duración", f"{frame_count/fps:.1f}s")
    
    # Controles
    st.write("---")
    
    # Placeholders
    placeholder_video = st.empty()
    placeholder_info = st.empty()
    placeholder_metrics = st.empty()
    
    # Botón de inicio
    if st.button("▶️ Procesar Video"):
        # Reiniciar buffer
        buffer.clear()
        
        # Métricas de seguimiento
        exercise_counts = {label: 0 for label in le.classes_}
        predictions_history = []
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_text.success("✅ Video procesado completamente")
                break
            
            frame_idx += 1
            progress = frame_idx / frame_count
            progress_bar.progress(progress)
            status_text.text(f"Procesando frame {frame_idx}/{frame_count}")
            
            # Procesa el frame con YOLOv8s-pose
            results = model_pose(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Extrae keypoints
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            predicted_label = "Esperando datos..."
            confidence = 0.0
            angles_text = ""
            
            if len(keypoints) > 0:
                keypoints = keypoints[0]  # Primera persona
                frame_height, frame_width = frame.shape[:2]
                keypoints_norm = keypoints / np.array([frame_width, frame_height])
                
                # Verifica que haya suficientes keypoints detectados
                if len(keypoints_norm) >= 17:
                    # Calcula ángulos
                    left_shoulder = keypoints_norm[5]
                    left_elbow = keypoints_norm[7]
                    left_wrist = keypoints_norm[9]
                    right_shoulder = keypoints_norm[6]
                    right_elbow = keypoints_norm[8]
                    right_wrist = keypoints_norm[10]
                    left_hip = keypoints_norm[11]
                    left_knee = keypoints_norm[13]
                    left_ankle = keypoints_norm[15]
                    right_hip = keypoints_norm[12]
                    right_knee = keypoints_norm[14]
                    right_ankle = keypoints_norm[16]
                    
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    
                    angles_text = f"Ángulos: L.Codo={left_elbow_angle:.1f}° R.Codo={right_elbow_angle:.1f}° L.Rodilla={left_knee_angle:.1f}° R.Rodilla={right_knee_angle:.1f}°"
                    
                    # Forma el frame_data (13 keypoints * 2 coords + 4 ángulos = 30 features)
                    keypoint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]
                    selected_keypoints = keypoints_norm[keypoint_indices].flatten()
                    frame_data = np.concatenate((selected_keypoints, 
                                                [right_elbow_angle, left_elbow_angle, 
                                                 right_knee_angle, left_knee_angle]))
                    
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
                            
                            # Registrar predicción
                            predictions_history.append({
                                'frame': frame_idx,
                                'exercise': predicted_label,
                                'confidence': confidence
                            })
                            
                            # Contar ejercicios
                            exercise_counts[predicted_label] += 1
                            
                            # Añadir texto al frame
                            cv2.putText(annotated_frame, f"{predicted_label}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Conf: {confidence:.1f}%", 
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
                            if show_angles:
                                y_pos = 110
                                cv2.putText(annotated_frame, f"L.Codo: {left_elbow_angle:.0f}°", 
                                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                cv2.putText(annotated_frame, f"R.Codo: {right_elbow_angle:.0f}°", 
                                          (10, y_pos+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                cv2.putText(annotated_frame, f"L.Rodilla: {left_knee_angle:.0f}°", 
                                          (10, y_pos+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                cv2.putText(annotated_frame, f"R.Rodilla: {right_knee_angle:.0f}°", 
                                          (10, y_pos+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Muestra en Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            with placeholder_video.container():
                st.image(img_pil, caption=f"Frame {frame_idx}/{frame_count}", use_column_width=True)
            
            # Información de predicción
            with placeholder_info.container():
                if len(buffer) < seq_length:
                    st.info(f"📊 Recolectando frames: {len(buffer)}/{seq_length}")
                else:
                    if confidence >= confidence_threshold:
                        if 'down' in predicted_label.lower():
                            st.info(f"⬇️ **{predicted_label}** (Confianza: {confidence:.1f}%)")
                        elif 'up' in predicted_label.lower():
                            st.success(f"⬆️ **{predicted_label}** (Confianza: {confidence:.1f}%)")
                        else:
                            st.write(f"🏋️ **{predicted_label}** (Confianza: {confidence:.1f}%)")
                    else:
                        st.warning(f"⚠️ Baja confianza: {confidence:.1f}% - Verifica postura")
                
                if show_angles and angles_text:
                    st.caption(angles_text)
            
            # Métricas en tiempo real
            with placeholder_metrics.container():
                cols = st.columns(len(le.classes_))
                for idx, (label, count) in enumerate(exercise_counts.items()):
                    cols[idx].metric(label.replace('_', ' ').title(), count)
        
        # Resumen final
        st.write("---")
        st.subheader("📊 Resumen del Análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Detecciones por ejercicio:**")
            for label, count in exercise_counts.items():
                percentage = (count / frame_count) * 100 if frame_count > 0 else 0
                st.write(f"- {label.replace('_', ' ').title()}: {count} frames ({percentage:.1f}%)")
        
        with col2:
            if predictions_history:
                st.write("**Estadísticas de confianza:**")
                confidences = [p['confidence'] for p in predictions_history]
                st.write(f"- Confianza promedio: {np.mean(confidences):.1f}%")
                st.write(f"- Confianza mínima: {np.min(confidences):.1f}%")
                st.write(f"- Confianza máxima: {np.max(confidences):.1f}%")
        
        progress_bar.empty()
        status_text.empty()
    
    cap.release()
    
    # Limpia el archivo temporal
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

else:
    st.info("👆 Por favor, carga un video para comenzar el análisis")
    
    # Ejemplos de ejercicios
    st.write("---")
    st.subheader("📋 Ejercicios Detectables")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Push-ups:**")
        st.write("- pushups_down")
        st.write("- pushups_up")
    
    with col2:
        st.write("**Sit-ups:**")
        st.write("- situp_down")
        st.write("- situp_up")
    
    with col3:
        st.write("**Squats:**")
        st.write("- squats_down")
        st.write("- squats_up")