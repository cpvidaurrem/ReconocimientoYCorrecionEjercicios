"""
API Backend para Aplicación Móvil de Evaluación de Ejercicios
Usa FastAPI para servir el modelo LSTM en tiempo real
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
import base64
import io
from PIL import Image
import sys
import os
from typing import List, Dict
from pydantic import BaseModel

# Configuración del proyecto
PROJECT_DIR = r'C:\Users\CHRISTIAN\Documents\EvaluacionEjerciciosIA'
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

from utils import calculate_angle

# Inicializar FastAPI
app = FastAPI(
    title="Exercise Evaluation API",
    description="API para evaluación de ejercicios en tiempo real",
    version="1.0.0"
)

# Configurar CORS (permitir peticiones desde la app móvil)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 10

# Modelo LSTM
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, 
                                   dropout=dropout if num_layers > 1 else 0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Cargar modelos globalmente
print("Cargando modelos...")
model_pose = YOLO(os.path.join(PROJECT_DIR, 'models/yolov8s-pose.pt'))

model_lstm = LSTMModel(input_size=30, hidden_size=64, num_layers=2, num_classes=6, dropout=0.3).to(device)
checkpoint = torch.load(os.path.join(PROJECT_DIR, 'models/best_lstm_model.pth'), map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_lstm.load_state_dict(checkpoint['model_state_dict'])
else:
    model_lstm.load_state_dict(checkpoint)
model_lstm.eval()

# Label Encoder
le = LabelEncoder()
le.classes_ = np.array(['pushups_down', 'pushups_up', 'situp_down', 'situp_up', 'squats_down', 'squats_up'])

# Buffer de secuencias por sesión (diccionario por session_id)
session_buffers = {}

print(f"Modelos cargados correctamente en dispositivo: {device}")

# Modelos de datos
class PredictionResponse(BaseModel):
    success: bool
    exercise: str
    confidence: float
    angles: Dict[str, float]
    buffer_status: str
    keypoints_detected: bool
    message: str

class SessionInfo(BaseModel):
    session_id: str
    buffer_size: int
    device: str
    model_info: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz - información de la API"""
    return {
        "message": "Exercise Evaluation API",
        "version": "1.0.0",
        "status": "running",
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "predict_frame": "/predict/frame",
            "predict_base64": "/predict/base64",
            "reset_session": "/session/reset/{session_id}",
            "session_info": "/session/info/{session_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Verificar estado del servidor"""
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": True
    }

@app.post("/predict/frame", response_model=PredictionResponse)
async def predict_from_frame(
    file: UploadFile = File(...),
    session_id: str = "default"
):
    """
    Predice ejercicio desde una imagen (frame de video)
    
    Parameters:
    - file: Imagen en formato JPG/PNG
    - session_id: ID de sesión para mantener buffer de secuencias
    """
    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        # Procesar frame
        result = process_frame(frame, session_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando frame: {str(e)}")

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(
    image_base64: str,
    session_id: str = "default"
):
    """
    Predice ejercicio desde imagen en base64
    
    Parameters:
    - image_base64: Imagen codificada en base64
    - session_id: ID de sesión
    """
    try:
        # Decodificar base64
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen base64")
        
        # Procesar frame
        result = process_frame(frame, session_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando base64: {str(e)}")

@app.delete("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """Resetea el buffer de una sesión"""
    if session_id in session_buffers:
        del session_buffers[session_id]
    return {
        "success": True,
        "message": f"Sesión {session_id} reseteada",
        "session_id": session_id
    }

@app.get("/session/info/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Obtiene información de una sesión"""
    buffer_size = len(session_buffers.get(session_id, []))
    return SessionInfo(
        session_id=session_id,
        buffer_size=buffer_size,
        device=str(device),
        model_info=f"LSTM (hidden={model_lstm.hidden_size}, layers={model_lstm.num_layers})"
    )

# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def process_frame(frame: np.ndarray, session_id: str) -> PredictionResponse:
    """
    Procesa un frame y retorna predicción
    
    Args:
        frame: Frame de video (numpy array BGR)
        session_id: ID de sesión para buffer
    
    Returns:
        PredictionResponse con predicción y metadatos
    """
    # Inicializar buffer si no existe
    if session_id not in session_buffers:
        session_buffers[session_id] = []
    
    buffer = session_buffers[session_id]
    
    # Detectar pose con YOLOv8
    results = model_pose(frame, verbose=False)
    keypoints = results[0].keypoints.xy.cpu().numpy()
    
    # Valores por defecto
    predicted_label = "waiting"
    confidence = 0.0
    angles = {"left_elbow": 0, "right_elbow": 0, "left_knee": 0, "right_knee": 0}
    keypoints_detected = False
    message = "Esperando detección de persona"
    
    if len(keypoints) > 0 and len(keypoints[0]) >= 17:
        keypoints_detected = True
        keypoints = keypoints[0]
        
        # Normalizar keypoints
        frame_height, frame_width = frame.shape[:2]
        keypoints_norm = keypoints / np.array([frame_width, frame_height])
        
        # Calcular ángulos
        try:
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
            
            angles = {
                "left_elbow": float(left_elbow_angle),
                "right_elbow": float(right_elbow_angle),
                "left_knee": float(left_knee_angle),
                "right_knee": float(right_knee_angle)
            }
            
            # Crear feature vector (30 features)
            keypoint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            selected_keypoints = keypoints_norm[keypoint_indices].flatten()
            frame_data = np.concatenate((selected_keypoints, 
                                        [right_elbow_angle, left_elbow_angle, 
                                         right_knee_angle, left_knee_angle]))
            
            # Añadir al buffer
            buffer.append(frame_data)
            if len(buffer) > SEQUENCE_LENGTH:
                buffer.pop(0)
            
            # Actualizar buffer en diccionario
            session_buffers[session_id] = buffer
            
            # Predecir si buffer está lleno
            if len(buffer) == SEQUENCE_LENGTH:
                sequence = np.array(buffer)
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model_lstm(sequence)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    predicted_label = le.inverse_transform([predicted.cpu().numpy()])[0]
                    confidence = float(probabilities[0][predicted.cpu().numpy()].item() * 100)
                
                message = "Predicción activa"
            else:
                message = f"Recolectando frames: {len(buffer)}/{SEQUENCE_LENGTH}"
        
        except Exception as e:
            message = f"Error calculando ángulos: {str(e)}"
    
    # Determinar estado del buffer
    buffer_status = f"{len(buffer)}/{SEQUENCE_LENGTH}"
    
    return PredictionResponse(
        success=True,
        exercise=predicted_label,
        confidence=confidence,
        angles=angles,
        buffer_status=buffer_status,
        keypoints_detected=keypoints_detected,
        message=message
    )

# ============================================================================
# INICIAR SERVIDOR
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 INICIANDO API DE EVALUACIÓN DE EJERCICIOS")
    print("="*70)
    print(f"Dispositivo: {device}")
    print(f"Modelos cargados: ✓")
    print(f"Puerto: 8000")
    print(f"Acceso local: http://localhost:8000")
    print(f"Acceso red local: http://192.168.0.11:8000")
    print(f"Documentación: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_mobile:app",
        host="0.0.0.0",  # Accesible desde red local
        port=8000,
        reload=True  # Auto-reload en desarrollo
    )