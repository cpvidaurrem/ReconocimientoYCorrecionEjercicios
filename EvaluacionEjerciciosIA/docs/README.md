# Evaluación Automática de Ejercicios en Educación Física con IA
Proyecto para reconocer y evaluar ejercicios (sentadillas, flexiones, planchas, abdominales) usando visión por computadora e IA.
- Entorno: Python 3.10/3.11 con PyTorch 2.8.0+cu126 y CUDA 12.6.
- Instrucciones: Activa el entorno virtual (.venv\Scripts\activate en Windows) y abre notebooks en VS Code.

# Para ejecutar Prototipo
streamlit run notebooks/prototipo_videoKey.py --server.enableCORS false

# Para ejecutar servidor para la app

cd notebooks
python api_mobile.py