// import 'dart:async';
// import 'dart:io';
// import 'package:camera/camera.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter_tts/flutter_tts.dart';
// import 'package:http/http.dart' as http;
// import 'package:path_provider/path_provider.dart';
// import 'package:permission_handler/permission_handler.dart';
// import 'dart:convert';

// List<CameraDescription> cameras = [];

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   cameras = await availableCameras();
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       home: CameraScreen(),
//     );
//   }
// }

// class CameraScreen extends StatefulWidget {
//   const CameraScreen({super.key});

//   @override
//   State<CameraScreen> createState() => _CameraScreenState();
// }

// class _CameraScreenState extends State<CameraScreen> {
//   late CameraController _controller;
//   late Timer _timer;
//   final FlutterTts _flutterTts = FlutterTts();

//   @override
//   void initState() {
//     super.initState();
//     _initPermissions();
//   }

//   Future<void> _initPermissions() async {
//     await [Permission.camera, Permission.microphone, Permission.storage]
//         .request();
//     _initializeCamera();
//   }

//   Future<void> _initializeCamera() async {
//     _controller = CameraController(cameras[0], ResolutionPreset.medium);
//     await _controller.initialize();
//     if (!mounted) return;
//     setState(() {});
//     _startFrameCapture();
//   }

//   void _startFrameCapture() {
//     _timer =
//         Timer.periodic(const Duration(seconds: 5), (_) => _captureAndSend());
//   }

//   Future<void> _captureAndSend() async {
//     if (!_controller.value.isInitialized || _controller.value.isTakingPicture)
//       return;

//     try {
//       final XFile file = await _controller.takePicture();
//       final File imageFile = File(file.path);
//       final uri = Uri.parse('http://172.20.10.4:5001/predict');
//       final request = http.MultipartRequest('POST', uri)
//         ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

//       final response = await request.send();
//       final respStr = await response.stream.bytesToString();
//       final decoded = json.decode(respStr);

//       if (decoded['caption'] != null) {
//         _speak(decoded['caption']);
//       }
//     } catch (e) {
//       debugPrint('Error sending frame: $e');
//     }
//   }

//   Future<void> _speak(String text) async {
//     await _flutterTts.stop();
//     await _flutterTts.speak(text);
//   }

//   @override
//   void dispose() {
//     _timer.cancel();
//     _controller.dispose();
//     _flutterTts.stop();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     if (!_controller.value.isInitialized) {
//       return const Scaffold(body: Center(child: CircularProgressIndicator()));
//     }
//     return Scaffold(
//       body: CameraPreview(_controller),
//     );
//   }
// }

// import 'dart:async';
// import 'dart:io';
// import 'dart:convert';
// import 'package:camera/camera.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter_tts/flutter_tts.dart';
// import 'package:http/http.dart' as http;
// import 'package:path_provider/path_provider.dart';
// import 'package:permission_handler/permission_handler.dart';

// List<CameraDescription> cameras = [];

// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   cameras = await availableCameras();
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return const MaterialApp(
//       debugShowCheckedModeBanner: false,
//       home: CameraScreen(),
//     );
//   }
// }

// class CameraScreen extends StatefulWidget {
//   const CameraScreen({super.key});

//   @override
//   State<CameraScreen> createState() => _CameraScreenState();
// }

// class _CameraScreenState extends State<CameraScreen> {
//   late CameraController _controller;
//   late Timer _timer;
//   final FlutterTts _flutterTts = FlutterTts();

//   @override
//   void initState() {
//     super.initState();
//     _initPermissions();
//   }

//   Future<void> _initPermissions() async {
//     await [Permission.camera, Permission.microphone, Permission.storage]
//         .request();
//     _initializeCamera();
//   }

//   Future<void> _initializeCamera() async {
//     _controller = CameraController(cameras[0], ResolutionPreset.medium);
//     await _controller.initialize();
//     if (!mounted) return;
//     setState(() {});
//     _startFrameCapture();
//   }

//   void _startFrameCapture() {
//     _timer =
//         Timer.periodic(const Duration(seconds: 5), (_) => _captureAndSend());
//   }

//   Future<void> _captureAndSend() async {
//     if (!_controller.value.isInitialized || _controller.value.isTakingPicture)
//       return;

//     try {
//       final XFile file = await _controller.takePicture();
//       final File imageFile = File(file.path);
//       final uri = Uri.parse('http://172.20.10.4:5001/predict'); // your local IP
//       final request = http.MultipartRequest('POST', uri)
//         ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

//       final response = await request.send();
//       final respStr = await response.stream.bytesToString();
//       final decoded = json.decode(respStr);

//       if (decoded['caption'] != null) {
//         _speak(decoded['caption']);
//       }
//     } catch (e) {
//       debugPrint('Error sending frame: $e');
//     }
//   }

//   Future<void> _speak(String text) async {
//     await _flutterTts.stop();
//     await _flutterTts.speak(text);
//   }

//   @override
//   void dispose() {
//     _timer.cancel();
//     _controller.dispose();
//     _flutterTts.stop();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     if (!_controller.value.isInitialized) {
//       return const Scaffold(body: Center(child: CircularProgressIndicator()));
//     }

//     return Scaffold(
//       body: SizedBox.expand(
//         child: CameraPreview(_controller),
//       ),
//     );
//   }
// }
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

// -------------------- HOME SCREEN --------------------

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late stt.SpeechToText _speech;
  bool _isListening = false;
  String lastWords = "";

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
    _requestPermissions();
    _initSpeech();
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.microphone,
      Permission.speech,
      Permission.camera,
      Permission.storage,
    ].request();
  }

  void _initSpeech() async {
    bool available = await _speech.initialize(
      onStatus: (status) => print(" Status: $status"),
      onError: (error) => print(" Error: $error"),
    );

    print(" Speech available: $available");

    if (available) {
      _startListening();
    } else {
      print(" Speech recognition not available");
    }
  }

  void _startListening() {
    if (!_isListening) {
      setState(() => _isListening = true);
      _speech.listen(
        onResult: (result) {
          final command = result.recognizedWords.toLowerCase().trim();
          print("Heard: $command");

          setState(() {
            lastWords = command;
          });

          if (command.contains("scan") && command.contains("product")) {
            _speech.stop();
            setState(() => _isListening = false);
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const CameraScreen()),
            );
          }
        },
        partialResults: true,
        listenMode: stt.ListenMode.confirmation,
        localeId: "en_US",
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: _isListening ? Colors.green : Colors.blueAccent,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 12,
                    spreadRadius: 2,
                  )
                ],
              ),
              child: const Icon(
                Icons.mic,
                color: Colors.white,
                size: 60,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              "Say 'scan product' to begin",
              style: TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 16),
            Text(
              "Heard: \"$lastWords\"",
              style: const TextStyle(fontSize: 14, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }
}

// -------------------- CAMERA SCREEN --------------------
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with TickerProviderStateMixin {
  late CameraController _controller;
  late Timer _timer;
  final FlutterTts _flutterTts = FlutterTts();
  final List<String> _captions = [];

  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 400),
      vsync: this,
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeIn,
    );
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(cameras[0], ResolutionPreset.high);
    await _controller.initialize();
    if (!mounted) return;
    setState(() {});
    _startFrameCapture();
  }

  void _startFrameCapture() {
    _timer =
        Timer.periodic(const Duration(seconds: 5), (_) => _captureAndSend());
  }

  Future<void> _captureAndSend() async {
    if (!_controller.value.isInitialized || _controller.value.isTakingPicture) {
      return;
    }

    try {
      final XFile file = await _controller.takePicture();
      final File imageFile = File(file.path);

      final uri = Uri.parse(
          'http://172.20.10.2:5001/predict'); // replace with your local IP
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

      final response = await request.send();
      final respStr = await response.stream.bytesToString();
      final decoded = json.decode(respStr);

      if (decoded['caption'] != null) {
        final result = decoded['caption'];
        setState(() {
          _captions.insert(0, result);
          if (_captions.length > 3) _captions.removeLast();
        });
        _fadeController.forward(from: 0);
        _speak(result);
      }
    } catch (e) {
      debugPrint(' Error sending frame: $e');
    }
  }

  Future<void> _speak(String text) async {
    await _flutterTts.stop();
    await _flutterTts.speak(text);
  }

  @override
  void dispose() {
    _timer.cancel();
    _controller.dispose();
    _flutterTts.stop();
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Fullscreen camera with 9:16 aspect ratio
          Center(
            child: AspectRatio(
              aspectRatio: 9 / 16,
              child: CameraPreview(_controller),
            ),
          ),
          // Caption overlay (bottom 25% of screen height)
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              height: screenHeight * 0.25,
              color: Colors.black.withOpacity(0.65),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: FadeTransition(
                opacity: _fadeAnimation,
                child: ListView.builder(
                  itemCount: _captions.length,
                  itemBuilder: (context, index) {
                    return Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.8),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Text(
                        _captions[index],
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
