# 📱 GUÍA COMPLETA: APLICACIÓN MÓVIL DE EVALUACIÓN DE EJERCICIOS

## 🎯 ARQUITECTURA DEL SISTEMA

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   APP MÓVIL     │ ◄─────► │   API BACKEND    │ ◄─────► │  MODELOS IA     │
│   (Flutter)     │  HTTP   │   (FastAPI)      │         │  (PyTorch/YOLO) │
└─────────────────┘         └──────────────────┘         └─────────────────┘
  - Captura video           - Procesa frames             - YOLOv8s-pose
  - Muestra resultados      - Detecta pose               - LSTM entrenado
  - Interfaz usuario        - Clasifica ejercicio        - Predicción
```

---

## 📦 PARTE 1: CONFIGURAR EL BACKEND (API)

### 1.1 Instalar dependencias

```bash
cd C:\Users\CHRISTIAN\Documents\EvaluacionEjerciciosIA

# Activar entorno virtual
.\.venv\Scripts\activate

# Instalar FastAPI y uvicorn
pip install fastapi uvicorn[standard] python-multipart
```

### 1.2 Crear el archivo API

Guarda el código `api_mobile.py` en la carpeta del proyecto:
```
EvaluacionEjerciciosIA/
├── notebooks/
│   └── api_mobile.py  ← AQUÍ
├── models/
├── src/
└── ...
```

### 1.3 Obtener tu IP local

**Windows:**
```bash
ipconfig
# Busca: IPv4 Address. . . . . : 192.168.X.X
```

**Linux/Mac:**
```bash
ifconfig
# Busca: inet 192.168.X.X
```

### 1.4 Iniciar el servidor

```bash
cd notebooks
python api_mobile.py
```

Deberías ver:
```
🚀 INICIANDO API DE EVALUACIÓN DE EJERCICIOS
Dispositivo: cuda
Modelos cargados: ✓
Puerto: 8000
Acceso red local: http://192.168.X.X:8000
```

### 1.5 Probar la API

Abre el navegador:
```
http://localhost:8000/docs
```

Verás la documentación interactiva (Swagger UI).

**Prueba rápida:**
```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "models_loaded": true
}
```

---

## 📱 PARTE 2: CONFIGURAR LA APP FLUTTER

### 2.1 Instalar Flutter

**Windows:**
1. Descarga Flutter SDK: https://docs.flutter.dev/get-started/install/windows
2. Extrae en `C:\src\flutter`
3. Agrega al PATH: `C:\src\flutter\bin`
4. Verifica: `flutter doctor`

**Problemas comunes:**
- ✅ Instala Android Studio
- ✅ Acepta licencias: `flutter doctor --android-licenses`
- ✅ Instala Visual Studio Code con extensión Flutter

### 2.2 Crear proyecto Flutter

```bash
# Crear nuevo proyecto
flutter create exercise_evaluator
cd exercise_evaluator
```

### 2.3 Configurar archivos

**Reemplaza `lib/main.dart`:**
```bash
# Copia el contenido del artifact "main.dart" completo
```

**Reemplaza `pubspec.yaml`:**
```bash
# Copia el contenido del artifact "pubspec.yaml"
```

**Instala dependencias:**
```bash
flutter pub get
```

### 2.4 Configurar permisos

#### **Android** (`android/app/src/main/AndroidManifest.xml`):

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <!-- Agregar ANTES de <application> -->
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    
    <uses-feature android:name="android.hardware.camera" />
    <uses-feature android:name="android.hardware.camera.autofocus" />
    
    <application ...>
        
        <!-- Agregar dentro de <application> -->
        <meta-data
            android:name="android.permission.CAMERA"
            android:value="true" />
        
        <!-- Permitir HTTP (desarrollo) -->
        <meta-data
            android:name="android.security.network_security_config"
            android:resource="@xml/network_security_config" />
    </application>
</manifest>
```

#### **Crear archivo de configuración de red:**

`android/app/src/main/res/xml/network_security_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="true">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

#### **iOS** (`ios/Runner/Info.plist`):

```xml
<dict>
    <!-- Agregar estas líneas -->
    <key>NSCameraUsageDescription</key>
    <string>Se necesita acceso a la cámara para evaluar ejercicios</string>
    
    <key>NSMicrophoneUsageDescription</key>
    <string>No se usa el micrófono</string>
    
    <key>NSPhotoLibraryUsageDescription</key>
    <string>Acceso opcional a la galería</string>
</dict>
```

### 2.5 Configurar la IP del API

En `lib/main.dart`, línea ~37:

```dart
String apiUrl = "http://192.168.1.100:8000"; // ⚠️ CAMBIA POR TU IP
```

**Reemplaza `192.168.1.100` por tu IP local del paso 1.3**

---

## 🚀 PARTE 3: EJECUTAR LA APLICACIÓN

### 3.1 Conectar dispositivo

**Opción A: Dispositivo físico (RECOMENDADO)**

1. Habilita "Modo Desarrollador" en tu teléfono:
   - Android: Configuración → Acerca del teléfono → Toca "Número de compilación" 7 veces
   - Habilita "Depuración USB"

2. Conecta el teléfono por USB

3. Verifica:
```bash
flutter devices
```

Deberías ver tu dispositivo listado.

**Opción B: Emulador Android**

```bash
# Crear emulador
flutter emulators --create

# Iniciar emulador
flutter emulators --launch <emulator_id>
```

⚠️ **NOTA:** El emulador debe estar en la misma red que tu PC.

### 3.2 Ejecutar la app

```bash
flutter run
```

O en VS Code:
- Presiona `F5`
- O click en "Run" → "Start Debugging"

### 3.3 Usar la aplicación

1. **Permite permisos** de cámara cuando se solicite
2. **Configura la URL del API** (ícono ⚙️ arriba a la derecha)
3. **Presiona ▶️** (botón verde) para iniciar
4. **Observa** la predicción en tiempo real
5. **Presiona ⏹️** (botón rojo) para detener
6. **Presiona 🔄** (botón naranja) para resetear estadísticas

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### Problema 1: "Error de conexión con API"

✅ **Solución:**
- Verifica que el servidor API esté corriendo: `http://localhost:8000/health`
- Asegúrate de estar en la misma red WiFi
- Desactiva firewall temporalmente
- Verifica la IP configurada en la app

**Probar conexión desde el móvil:**
- Abre navegador en el móvil
- Visita: `http://TU_IP:8000/health`
- Deberías ver la respuesta JSON

### Problema 2: "No hay cámaras disponibles"

✅ **Solución:**
- Verifica permisos en Configuración del dispositivo
- Desinstala y reinstala la app
- Usa dispositivo físico en lugar de emulador

### Problema 3: "Esperando detección de persona"

✅ **Solución:**
- Asegúrate de tener buena iluminación
- Coloca la persona completa en el cuadro
- Mantén distancia adecuada (2-3 metros)
- Verifica que YOLOv8 esté detectando en el servidor

### Problema 4: Latencia alta / App lenta

✅ **Solución:**
- Reduce la frecuencia de frames (en `main.dart`, línea ~119):
  ```dart
  Duration(milliseconds: 1000) // De 500ms a 1000ms (1 FPS)
  ```
- Usa `ResolutionPreset.low` en lugar de `medium`
- Verifica que el servidor esté usando GPU

### Problema 5: "Gradle build failed" (Android)

✅ **Solución:**
```bash
cd android
./gradlew clean
cd ..
flutter clean
flutter pub get
flutter run
```

---

## ⚡ OPTIMIZACIONES OPCIONALES

### Opción 1: Reducir uso de datos

En `main.dart`, cambiar calidad de imagen:

```dart
_cameraController = CameraController(
  camera,
  ResolutionPreset.low, // low en lugar de medium
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.jpeg,
);
```

### Opción 2: Procesamiento local (sin servidor)

Para esto necesitas:
1. Convertir modelo PyTorch → TFLite
2. Usar plugin `tflite_flutter`
3. Mayor complejidad de implementación

**Script de conversión (para el futuro):**
```python
import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare

# PyTorch → ONNX
model.eval()
dummy_input = torch.randn(1, 10, 30)
torch.onnx.export(model, dummy_input, "model.onnx")

# ONNX → TensorFlow → TFLite
# (requiere más pasos)
```

### Opción 3: Agregar grabación de video

```dart
// En _ExerciseEvaluatorHomeState
import 'package:video_player/video_player.dart';

// Botón para iniciar grabación
FloatingActionButton(
  onPressed: () async {
    await _cameraController!.startVideoRecording();
  },
  child: Icon(Icons.videocam),
)
```

---

## 📊 CARACTERÍSTICAS ACTUALES

✅ Detección de pose en tiempo real  
✅ Clasificación de 6 tipos de ejercicios  
✅ Cálculo de ángulos articulares  
✅ Indicador de confianza  
✅ Estadísticas de ejercicios  
✅ Buffer de secuencias  
✅ Interfaz intuitiva  
✅ Modo oscuro  

---

## 🎯 MEJORAS FUTURAS

🔄 Modo offline (TFLite)  
🔄 Grabación y exportación de videos  
🔄 Historial de sesiones  
🔄 Gráficas de progreso  
🔄 Comparación con ejercicio correcto  
🔄 Contador automático de repeticiones  
🔄 Feedback de voz  
🔄 Múltiples usuarios  

---

## 📝 CHECKLIST DE IMPLEMENTACIÓN

### Backend:
- [ ] Instalar FastAPI
- [ ] Copiar `api_mobile.py`
- [ ] Obtener IP local
- [ ] Iniciar servidor
- [ ] Probar endpoint `/health`

### App Móvil:
- [ ] Instalar Flutter
- [ ] Crear proyecto
- [ ] Copiar `main.dart` y `pubspec.yaml`
- [ ] Configurar permisos (Android/iOS)
- [ ] Configurar IP del API
- [ ] Ejecutar `flutter pub get`
- [ ] Conectar dispositivo
- [ ] Ejecutar app

### Pruebas:
- [ ] Verificar conexión API
- [ ] Probar detección de persona
- [ ] Verificar predicciones
- [ ] Revisar ángulos articulares
- [ ] Validar estadísticas

---

## 🆘 SOPORTE

Si tienes problemas:

1. **Revisa logs del servidor:**
   ```bash
   # En la terminal donde corre el API
   # Verás requests entrantes y errores
   ```

2. **Revisa logs de Flutter:**
   ```bash
   flutter logs
   ```

3. **Debug en VS Code:**
   - Coloca breakpoints
   - Usa el inspector de variables
   - Revisa la consola de depuración

---

## 📚 RECURSOS ADICIONALES

- Flutter Docs: https://docs.flutter.dev/
- FastAPI Docs: https://fastapi.tiangolo.com/
- Camera Plugin: https://pub.dev/packages/camera
- HTTP Package: https://pub.dev/packages/http

---

¡Listo! Ahora tienes una aplicación móvil completa para evaluación de ejercicios en tiempo real. 🎉