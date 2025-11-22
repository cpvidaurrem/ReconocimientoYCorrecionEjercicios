"""
API Backend para Aplicación Móvil de Evaluación de Ejercicios
✨ NUEVA VERSIÓN: Incluye coordenadas de keypoints para feedback visual
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
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, ConfigDict

# Configuración del proyecto
PROJECT_DIR = r'C:\Users\CHRISTIAN\Documents\EvaluacionEjerciciosIA'
sys.path.append(os.path.join(PROJECT_DIR, 'src'))

from utils import calculate_angle

# Inicializar FastAPI
app = FastAPI(
    title="Exercise Evaluation API",
    description="API para evaluación de ejercicios en tiempo real con feedback visual",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración global con límite de memoria
import gc
import torch.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 10

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)

# ============================================================================ 
# MODELOS DE DATOS CORREGIDOS PARA Pydantic v2
# ============================================================================ 

class Keypoint(BaseModel):
    """Representa un keypoint con coordenadas y confianza"""
    x: float
    y: float
    confidence: float
    name: str

    model_config = ConfigDict(
        json_encoders={
            float: lambda v: round(v, 4)
        }
    )

class FormFeedback(BaseModel):
    """Feedback específico sobre la forma del ejercicio"""
    status: str
    message: str
    affected_joints: List[str]

class PredictionResponse(BaseModel):
    success: bool
    exercise: str
    confidence: float
    angles: Dict[str, float]
    keypoints: List[Dict[str, Any]]  # CORREGIDO: any → Any
    buffer_status: str
    keypoints_detected: bool
    message: str
    form_feedback: Optional[Dict[str, Any]] = None
    frame_dimensions: Dict[str, int]

    model_config = ConfigDict(
        json_encoders={
            float: lambda v: round(v, 4)
        }
    )

# ============================================================================ 
# MODELO LSTM
# ============================================================================ 

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

# ============================================================================ 
# CARGAR MODELOS
# ============================================================================ 

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

# Buffer de secuencias por sesión
session_buffers = {}

# Nombres de keypoints COCO
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

print(f"Modelos cargados correctamente en dispositivo: {device}")

# ============================================================================ 
# FUNCIONES DE ANÁLISIS DE FORMA
# ============================================================================ 

def analyze_exercise_form(exercise: str, angles: Dict[str, float], keypoints_norm: np.ndarray) -> FormFeedback:
    feedback = FormFeedback(
        status="good",
        message="Forma correcta",
        affected_joints=[]
    )
    
    # Squats
    if 'squats' in exercise:
        avg_knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
        if avg_knee_angle < 70:
            feedback.status = "warning"
            feedback.message = "⚠️ Rodillas muy flexionadas - riesgo de lesión"
            feedback.affected_joints = ["left_knee", "right_knee"]
        elif avg_knee_angle > 110 and 'down' in exercise:
            feedback.status = "warning"
            feedback.message = "⚠️ Baja más - sentadilla incompleta"
            feedback.affected_joints = ["left_knee", "right_knee"]
        
        left_hip = keypoints_norm[11]
        left_shoulder = keypoints_norm[5]
        if left_hip[1] > left_shoulder[1] + 0.3:
            feedback.status = "error"
            feedback.message = "❌ Mantén la espalda recta"
            feedback.affected_joints.extend(["left_hip", "left_shoulder"])
    
    # Pushups
    elif 'pushups' in exercise:
        avg_elbow_angle = (angles['left_elbow'] + angles['right_elbow']) / 2
        if avg_elbow_angle < 70:
            feedback.status = "warning"
            feedback.message = "⚠️ Codos muy flexionados"
            feedback.affected_joints = ["left_elbow", "right_elbow"]
        elif avg_elbow_angle > 140 and 'down' in exercise:
            feedback.status = "warning"
            feedback.message = "⚠️ Baja más - flexión incompleta"
            feedback.affected_joints = ["left_elbow", "right_elbow"]
        
        shoulder_y = (keypoints_norm[5][1] + keypoints_norm[6][1]) / 2
        hip_y = (keypoints_norm[11][1] + keypoints_norm[12][1]) / 2
        if abs(shoulder_y - hip_y) > 0.15:
            feedback.status = "error"
            feedback.message = "❌ Mantén el cuerpo recto"
            feedback.affected_joints.extend(["left_hip", "right_hip"])
    
    # Situps
    elif 'situp' in exercise:
        avg_knee_angle = (angles['left_knee'] + angles['right_knee']) / 2
        if avg_knee_angle > 100:
            feedback.status = "warning"
            feedback.message = "⚠️ Flexiona más las rodillas"
            feedback.affected_joints = ["left_knee", "right_knee"]
    
    if feedback.status == "good" and all(70 <= angle <= 170 for angle in angles.values()):
        feedback.status = "excellent"
        feedback.message = "✅ ¡Forma perfecta!"
    
    return feedback

# ============================================================================ 
# PROCESAMIENTO DE FRAMES
# ============================================================================ 

def process_frame(frame: np.ndarray, session_id: str) -> PredictionResponse:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if session_id not in session_buffers:
        session_buffers[session_id] = []
    buffer = session_buffers[session_id]
    frame_height, frame_width = frame.shape[:2]
    
    with torch.no_grad():
        results = model_pose(frame, verbose=False)
    keypoints_raw = results[0].keypoints.xy.cpu().numpy()
    confidences = results[0].keypoints.conf.cpu().numpy()
    
    predicted_label = "waiting"
    confidence = 0.0
    angles = {"left_elbow": 0, "right_elbow": 0, "left_knee": 0, "right_knee": 0}
    keypoints_list = []
    keypoints_detected = False
    message = "Esperando detección de persona"
    form_feedback_dict = None
    
    if len(keypoints_raw) > 0 and len(keypoints_raw[0]) >= 17:
        keypoints_detected = True
        keypoints = keypoints_raw[0]
        confs = confidences[0] if len(confidences) > 0 else np.ones(17)
        keypoints_norm = keypoints / np.array([frame_width, frame_height])
        
        for i, (kp, conf, name) in enumerate(zip(keypoints_norm, confs, KEYPOINT_NAMES)):
            keypoints_list.append({
                "x": float(kp[0]),
                "y": float(kp[1]),
                "confidence": float(conf),
                "name": name
            })
        
        try:
            left_shoulder, left_elbow, left_wrist = keypoints_norm[5], keypoints_norm[7], keypoints_norm[9]
            right_shoulder, right_elbow, right_wrist = keypoints_norm[6], keypoints_norm[8], keypoints_norm[10]
            left_hip, left_knee, left_ankle = keypoints_norm[11], keypoints_norm[13], keypoints_norm[15]
            right_hip, right_knee, right_ankle = keypoints_norm[12], keypoints_norm[14], keypoints_norm[16]
            
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
            
            keypoint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12]
            selected_keypoints = keypoints_norm[keypoint_indices].flatten()
            frame_data = np.concatenate((selected_keypoints, 
                                         [right_elbow_angle, left_elbow_angle, 
                                          right_knee_angle, left_knee_angle]))
            
            buffer.append(frame_data)
            if len(buffer) > SEQUENCE_LENGTH:
                buffer.pop(0)
            
            session_buffers[session_id] = buffer
            
            if len(buffer) == SEQUENCE_LENGTH:
                sequence = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model_lstm(sequence)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    predicted_label = le.inverse_transform([predicted.cpu().numpy()])[0]
                    confidence = float(probabilities[0][predicted.cpu().numpy()].item() * 100)
                
                form_feedback = analyze_exercise_form(predicted_label, angles, keypoints_norm)
                form_feedback_dict = {
                    "status": form_feedback.status,
                    "message": form_feedback.message,
                    "affected_joints": form_feedback.affected_joints
                }
                message = form_feedback.message
            else:
                message = f"Recolectando frames: {len(buffer)}/{SEQUENCE_LENGTH}"
                form_feedback_dict = None
        
        except Exception as e:
            message = f"Error calculando ángulos: {str(e)}"
    
    buffer_status = f"{len(buffer)}/{SEQUENCE_LENGTH}"
    
    return PredictionResponse(
        success=True,
        exercise=predicted_label,
        confidence=confidence,
        angles=angles,
        keypoints=keypoints_list,
        buffer_status=buffer_status,
        keypoints_detected=keypoints_detected,
        message=message,
        form_feedback=form_feedback_dict,
        frame_dimensions={"width": frame_width, "height": frame_height}
    )

# ============================================================================ 
# ENDPOINTS
# ============================================================================ 

@app.get("/")
async def root():
    return {
        "message": "Exercise Evaluation API v2.0",
        "version": "2.0.0",
        "status": "running",
        "device": str(device),
        "features": ["keypoint_tracking", "form_feedback", "visual_overlay"],
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
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        result = process_frame(frame, session_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando frame: {str(e)}")

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(
    image_base64: str,
    session_id: str = "default"
):
    try:
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen base64")
        
        result = process_frame(frame, session_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando base64: {str(e)}")

@app.delete("/session/reset/{session_id}")
async def reset_session(session_id: str):
    if session_id in session_buffers:
        del session_buffers[session_id]
    return {
        "success": True,
        "message": f"Sesión {session_id} reseteada",
        "session_id": session_id
    }

@app.get("/session/info/{session_id}")
async def get_session_info(session_id: str):
    buffer_size = len(session_buffers.get(session_id, []))
    return {
        "session_id": session_id,
        "buffer_size": buffer_size,
        "device": str(device),
        "model_info": f"LSTM (hidden={model_lstm.hidden_size}, layers={model_lstm.num_layers})"
    }

# ============================================================================ 
# INICIAR SERVIDOR
# ============================================================================ 

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 INICIANDO API DE EVALUACIÓN DE EJERCICIOS v2.0")
    print("="*70)
    print(f"Dispositivo: {device}")
    print(f"Modelos cargados: ✓")
    print(f"Características: Keypoint tracking, Form feedback, Visual overlay")
    print(f"Puerto: 8000")
    print(f"Acceso local: http://localhost:8000")
    print(f"Acceso red local: http://172.20.10.2:8000")
    print(f"Documentación: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_mobileV2:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
