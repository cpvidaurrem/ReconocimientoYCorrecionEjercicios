// ============================================================================
// APLICACIÓN FLUTTER - EVALUACIÓN DE EJERCICIOS CON FEEDBACK VISUAL
// ✨ NUEVA VERSIÓN: Skeleton overlay + alertas visuales en tiempo real
// ============================================================================

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

// ============================================================================
// MODELOS DE DATOS
// ============================================================================

class Keypoint {
  final double x;
  final double y;
  final double confidence;
  final String name;

  Keypoint({
    required this.x,
    required this.y,
    required this.confidence,
    required this.name,
  });

  factory Keypoint.fromJson(Map<String, dynamic> json) {
    return Keypoint(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      confidence: (json['confidence'] as num).toDouble(),
      name: json['name'] as String,
    );
  }

  Offset toOffset(Size screenSize) {
    return Offset(x * screenSize.width, y * screenSize.height);
  }
}

class FormFeedback {
  final String status; // "excellent", "good", "warning", "error"
  final String message;
  final List<String> affectedJoints;

  FormFeedback({
    required this.status,
    required this.message,
    required this.affectedJoints,
  });

  factory FormFeedback.fromJson(Map<String, dynamic> json) {
    return FormFeedback(
      status: json['status'] as String,
      message: json['message'] as String,
      affectedJoints: List<String>.from(json['affected_joints'] ?? []),
    );
  }

  Color getColor() {
    switch (status) {
      case 'excellent':
        return Colors.green;
      case 'good':
        return Colors.blue;
      case 'warning':
        return Colors.orange;
      case 'error':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  IconData getIcon() {
    switch (status) {
      case 'excellent':
        return Icons.check_circle;
      case 'good':
        return Icons.thumb_up;
      case 'warning':
        return Icons.warning;
      case 'error':
        return Icons.error;
      default:
        return Icons.info;
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(ExerciseEvaluatorApp());
}

class ExerciseEvaluatorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Evaluador de Ejercicios',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
      ),
      home: ExerciseEvaluatorHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

// ============================================================================
// HOME SCREEN
// ============================================================================

class ExerciseEvaluatorHome extends StatefulWidget {
  @override
  _ExerciseEvaluatorHomeState createState() => _ExerciseEvaluatorHomeState();
}

class _ExerciseEvaluatorHomeState extends State<ExerciseEvaluatorHome> {
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isProcessing = false;
  Timer? _frameTimer;

  String apiUrl = "http://172.20.10.2:8000";
  String sessionId = "mobile_session_${DateTime.now().millisecondsSinceEpoch}";

  // Estado de la predicción
  String currentExercise = "Esperando...";
  double confidence = 0.0;
  Map<String, double> angles = {
    "left_elbow": 0,
    "right_elbow": 0,
    "left_knee": 0,
    "right_knee": 0
  };
  List<Keypoint> keypoints = [];
  String bufferStatus = "0/10";
  bool keypointsDetected = false;
  String message = "Inicializando...";
  FormFeedback? formFeedback;

  // Dimensiones del frame para escalado
  Size frameDimensions = Size(640, 480);

  // Estadísticas
  int totalFramesProcessed = 0;
  int pushUpCount = 0;
  int sitUpCount = 0;
  int squatCount = 0;

  // Control de UI
  bool showSkeleton = true;
  bool showAngles = true;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _frameTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      setState(() {
        message = "No hay cámaras disponibles";
      });
      return;
    }

    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.low, // CAMBIADO: medium -> low para reducir carga
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      setState(() {
        _isCameraInitialized = true;
        message = "Cámara lista. Presiona Iniciar";
      });
    } catch (e) {
      setState(() {
        message = "Error inicializando cámara: $e";
      });
    }
  }

  void _startProcessing() {
    if (!_isCameraInitialized || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      message = "Procesando en tiempo real...";
    });

    // CRÍTICO: Reducir frecuencia a 1 FPS para evitar saturar GPU
    _frameTimer = Timer.periodic(Duration(milliseconds: 1000), (timer) async {
      if (_cameraController != null && _cameraController!.value.isInitialized) {
        await _captureAndProcess();
      }
    });
  }

  void _stopProcessing() {
    _frameTimer?.cancel();
    setState(() {
      _isProcessing = false;
      message = "Procesamiento detenido";
    });
  }

  Future<void> _captureAndProcess() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final XFile image = await _cameraController!.takePicture();
      final Uint8List imageBytes = await image.readAsBytes();

      // CRÍTICO: Comprimir imagen antes de enviar (reducir carga GPU)
      final compressed = await _compressImage(imageBytes);
      await _sendFrameToAPI(compressed);

      setState(() {
        totalFramesProcessed++;
      });
    } catch (e) {
      print("Error capturando frame: $e");
    }
  }

  // NUEVA FUNCIÓN: Comprimir imagen para reducir uso de memoria
  Future<Uint8List> _compressImage(Uint8List imageBytes) async {
    final codec = await ui.instantiateImageCodec(
      imageBytes,
      targetWidth: 480, // Reducir resolución a 480px ancho
    );
    final frame = await codec.getNextFrame();
    final byteData =
        await frame.image.toByteData(format: ui.ImageByteFormat.png);
    return byteData!.buffer.asUint8List();
  }

  Future<void> _sendFrameToAPI(Uint8List imageBytes) async {
    try {
      final uri = Uri.parse("$apiUrl/predict/frame?session_id=$sessionId");

      var request = http.MultipartRequest('POST', uri);
      request.files.add(
        http.MultipartFile.fromBytes('file', imageBytes, filename: 'frame.jpg'),
      );

      final response = await request.send().timeout(Duration(seconds: 5));

      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        final jsonData = json.decode(responseData);

        // DEBUG: Imprimir respuesta completa
        print("=== RESPUESTA API ===");
        print("Exercise: ${jsonData['exercise']}");
        print("Keypoints detected: ${jsonData['keypoints_detected']}");
        print("Keypoints count: ${jsonData['keypoints']?.length ?? 0}");
        print("Form feedback: ${jsonData['form_feedback']}");
        print("=====================");

        setState(() {
          currentExercise = jsonData['exercise'] ?? 'unknown';
          confidence = (jsonData['confidence'] ?? 0.0).toDouble();
          bufferStatus = jsonData['buffer_status'] ?? '0/10';
          keypointsDetected = jsonData['keypoints_detected'] ?? false;
          message = jsonData['message'] ?? 'Procesando...';

          // Parsear keypoints
          if (jsonData['keypoints'] != null) {
            keypoints = (jsonData['keypoints'] as List)
                .map((kp) => Keypoint.fromJson(kp))
                .toList();
            print("Keypoints parseados: ${keypoints.length}"); // DEBUG
          }

          // Parsear ángulos
          if (jsonData['angles'] != null) {
            angles = Map<String, double>.from(
              jsonData['angles'].map((k, v) => MapEntry(k, v.toDouble())),
            );
          }

          // Parsear feedback de forma
          if (jsonData['form_feedback'] != null) {
            formFeedback = FormFeedback.fromJson(jsonData['form_feedback']);
          }

          // Dimensiones del frame
          if (jsonData['frame_dimensions'] != null) {
            final dims = jsonData['frame_dimensions'];
            frameDimensions = Size(
              (dims['width'] as num).toDouble(),
              (dims['height'] as num).toDouble(),
            );
          }

          _updateExerciseCounts();
        });
      }
    } catch (e) {
      print("Error enviando frame: $e");
      setState(() {
        message = "Error de conexión con API";
      });
    }
  }

  void _updateExerciseCounts() {
    if (confidence < 80) return;

    if (currentExercise.contains('pushups'))
      pushUpCount++;
    else if (currentExercise.contains('situp'))
      sitUpCount++;
    else if (currentExercise.contains('squats')) squatCount++;
  }

  Future<void> _resetSession() async {
    try {
      final uri = Uri.parse("$apiUrl/session/reset/$sessionId");
      await http.delete(uri);

      setState(() {
        currentExercise = "Esperando...";
        confidence = 0.0;
        bufferStatus = "0/10";
        message = "Sesión reseteada";
        totalFramesProcessed = 0;
        pushUpCount = 0;
        sitUpCount = 0;
        squatCount = 0;
        keypoints = [];
        formFeedback = null;
      });
    } catch (e) {
      print("Error reseteando sesión: $e");
    }
  }

  // ============================================================================
  // UI COMPONENTS
  // ============================================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Evaluador de Ejercicios'),
        actions: [
          IconButton(
            icon: Icon(showSkeleton ? Icons.visibility : Icons.visibility_off),
            onPressed: () => setState(() => showSkeleton = !showSkeleton),
            tooltip: 'Toggle Skeleton',
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(flex: 3, child: _buildCameraPreview()),
          Expanded(flex: 2, child: _buildInfoPanel()),
        ],
      ),
      floatingActionButton: _buildControlButtons(),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraInitialized || _cameraController == null) {
      return const Center(child: CircularProgressIndicator());
    }

    print("=== BUILD CAMERA PREVIEW ===");
    print("showSkeleton: $showSkeleton");
    print("keypointsDetected: $keypointsDetected");
    print("keypoints.length: ${keypoints.length}");

    return Stack(
      children: [
        // Cámara
        Positioned.fill(child: CameraPreview(_cameraController!)),

        // Skeleton Overlay - SIEMPRE VISIBLE PARA DEBUG
        Positioned.fill(
          child: Container(
            color:
                Colors.transparent, // DEBUG: verificar que el container existe
            child: keypointsDetected && keypoints.length >= 17
                ? CustomPaint(
                    painter: SkeletonPainter(
                      keypoints: keypoints,
                      affectedJoints: formFeedback?.affectedJoints ?? [],
                      feedbackColor: formFeedback?.getColor() ?? Colors.blue,
                    ),
                  )
                : Center(
                    child: Text(
                      "Esperando keypoints...\nDetectados: ${keypoints.length}/17",
                      style: TextStyle(color: Colors.yellow, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                  ),
          ),
        ),

        // Info Overlay (arriba)
        Positioned(
          top: 16,
          left: 16,
          right: 16,
          child: _buildOverlayInfo(),
        ),

        // Feedback de forma (centro)
        if (formFeedback != null && _isProcessing)
          Positioned(
            left: 16,
            right: 16,
            bottom: 80,
            child: _buildFormFeedbackCard(),
          ),

        // Indicador de grabación
        if (_isProcessing)
          const Positioned(
            top: 16,
            right: 16,
            child: CircleAvatar(
              radius: 10,
              backgroundColor: Colors.red,
              child: Icon(Icons.fiber_manual_record,
                  size: 12, color: Colors.white),
            ),
          ),
      ],
    );
  }

  Widget _buildOverlayInfo() {
    Color confidenceColor = confidence >= 80
        ? Colors.green
        : confidence >= 60
            ? Colors.orange
            : Colors.red;

    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Wrap(
        spacing: 12,
        runSpacing: 4,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          Text(
            currentExercise.replaceAll('_', ' ').toUpperCase(),
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            'Confianza: ${confidence.toStringAsFixed(1)}%',
            style: TextStyle(
              color: confidenceColor,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            'Buffer: $bufferStatus',
            style: TextStyle(color: Colors.white70),
          ),
          if (keypointsDetected)
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.person, color: Colors.green, size: 14),
                SizedBox(width: 4),
                Text(
                  "Persona detectada",
                  style: TextStyle(color: Colors.green),
                ),
              ],
            ),
        ],
      ),
    );
  }

  Widget _buildFormFeedbackCard() {
    return AnimatedContainer(
      duration: Duration(milliseconds: 300),
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: formFeedback!.getColor().withOpacity(0.9),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.3),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(
            formFeedback!.getIcon(),
            color: Colors.white,
            size: 32,
          ),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              formFeedback!.message,
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoPanel() {
    return Container(
      color: Colors.grey[900],
      padding: const EdgeInsets.all(16),
      child: SingleChildScrollView(
        child: Column(
          children: [
            const Text(
              'Ángulos Articulares',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _buildAngleCard('Codo Izq', angles['left_elbow']!),
                  const SizedBox(width: 16),
                  _buildAngleCard('Codo Der', angles['right_elbow']!),
                  const SizedBox(width: 16),
                  _buildAngleCard('Rodilla Izq', angles['left_knee']!),
                  const SizedBox(width: 16),
                  _buildAngleCard('Rodilla Der', angles['right_knee']!),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildStatCard('Push-ups', pushUpCount, Icons.fitness_center),
                _buildStatCard('Sit-ups', sitUpCount, Icons.accessibility_new),
                _buildStatCard('Squats', squatCount, Icons.directions_run),
              ],
            ),
            const SizedBox(height: 20),
            Text(
              message,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.white70),
            ),
            const SizedBox(height: 8),
            Text(
              'Frames procesados: $totalFramesProcessed',
              style: const TextStyle(color: Colors.white54, fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAngleCard(String label, double angle) {
    return Column(
      children: [
        Text(label, style: const TextStyle(color: Colors.white70)),
        const SizedBox(height: 4),
        Text(
          '${angle.toStringAsFixed(0)}°',
          style: const TextStyle(
              color: Colors.blue, fontSize: 20, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildStatCard(String label, int count, IconData icon) {
    return Column(
      children: [
        Icon(icon, color: Colors.white, size: 24),
        Text(label,
            style: const TextStyle(color: Colors.white70, fontSize: 12)),
        Text(
          '$count',
          style: const TextStyle(
              color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildControlButtons() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        FloatingActionButton(
          heroTag: 'startstop',
          onPressed: _isProcessing ? _stopProcessing : _startProcessing,
          backgroundColor: _isProcessing ? Colors.red : Colors.green,
          child: Icon(_isProcessing ? Icons.stop : Icons.play_arrow),
        ),
        const SizedBox(height: 14),
        FloatingActionButton(
          heroTag: 'reset',
          onPressed: _resetSession,
          backgroundColor: Colors.orange,
          child: const Icon(Icons.refresh),
        ),
      ],
    );
  }

  void _showSettingsDialog() {
    final TextEditingController urlController =
        TextEditingController(text: apiUrl);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Configuración'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: urlController,
              decoration: const InputDecoration(labelText: 'URL del API'),
            ),
            SizedBox(height: 16),
            SwitchListTile(
              title: Text('Mostrar Skeleton'),
              value: showSkeleton,
              onChanged: (val) => setState(() => showSkeleton = val),
            ),
            SwitchListTile(
              title: Text('Mostrar Ángulos'),
              value: showAngles,
              onChanged: (val) => setState(() => showAngles = val),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancelar'),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() => apiUrl = urlController.text);
              Navigator.pop(context);
            },
            child: const Text('Guardar'),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// CUSTOM PAINTER - SKELETON OVERLAY
// ============================================================================

class SkeletonPainter extends CustomPainter {
  final List<Keypoint> keypoints;
  final List<String> affectedJoints;
  final Color feedbackColor;

  SkeletonPainter({
    required this.keypoints,
    required this.affectedJoints,
    required this.feedbackColor,
  });

  // Definir conexiones del skeleton (formato COCO)
  static const List<List<int>> SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Cabeza
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Brazos
    [5, 11], [6, 12], [11, 12], // Torso
    [11, 13], [13, 15], [12, 14], [14, 16], // Piernas
  ];

  @override
  void paint(Canvas canvas, Size size) {
    print("=== SKELETON PAINTER ===");
    print("Keypoints recibidos: ${keypoints.length}");
    print("Canvas size: $size");
    print("showSkeleton debería estar activo");

    if (keypoints.length < 17) {
      print("⚠️ Keypoints insuficientes: ${keypoints.length}/17");
      return;
    }

    // Pintar conexiones (líneas)
    final linePaint = Paint()
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    for (var connection in SKELETON_CONNECTIONS) {
      final kp1 = keypoints[connection[0]];
      final kp2 = keypoints[connection[1]];

      // Solo dibujar si ambos keypoints tienen confianza > 0.5
      if (kp1.confidence > 0.5 && kp2.confidence > 0.5) {
        final p1 = kp1.toOffset(size);
        final p2 = kp2.toOffset(size);

        // Cambiar color si la articulación está afectada
        bool isAffected = affectedJoints.contains(kp1.name) ||
            affectedJoints.contains(kp2.name);

        linePaint.color =
            isAffected ? feedbackColor : Colors.cyan.withOpacity(0.7);

        canvas.drawLine(p1, p2, linePaint);
      }
    }

    // Pintar keypoints (círculos)
    final pointPaint = Paint()..style = PaintingStyle.fill;

    for (var kp in keypoints) {
      if (kp.confidence > 0.5) {
        final point = kp.toOffset(size);

        // Color del punto según si está afectado
        bool isAffected = affectedJoints.contains(kp.name);
        pointPaint.color = isAffected ? feedbackColor : Colors.green;

        // Dibujar punto
        canvas.drawCircle(point, 6, pointPaint);

        // Borde blanco
        canvas.drawCircle(
          point,
          6,
          Paint()
            ..style = PaintingStyle.stroke
            ..color = Colors.white
            ..strokeWidth = 2,
        );
      }
    }
  }

  @override
  bool shouldRepaint(SkeletonPainter oldDelegate) {
    return keypoints != oldDelegate.keypoints ||
        affectedJoints != oldDelegate.affectedJoints ||
        feedbackColor != oldDelegate.feedbackColor;
  }
}
