import 'dart:developer';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_sudoku_app/camera_state.dart';
import 'package:native_opencv/native_opencv.dart';
import 'dart:io';

import 'package:provider/provider.dart';

void main() {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatelessWidget {
  final String title;
  OverlayEntry? entry;
  late File selectedImage;
  final picker = ImagePicker();

  MyHomePage({super.key, required this.title});

  // final nativeOpencv =

  final nativeOpenCV = NativeOpencv();

  @override
  Widget build(BuildContext context) {
    /* Jorge Chavez
    Description: This generates an overlay on the screen which provides
    the user with the option to upload an image or take a picture with their
    on device camera
    */
    void showOverlayOptions() {
      entry = OverlayEntry(
        builder: (context) => Stack(children: [
          ModalBarrier(
            onDismiss: () {
              entry?.remove();
            },
          ),
          Center(
            child: Container(
              decoration: BoxDecoration(
                color: Colors.blue,
                gradient: const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment(0.8, 1),
                  colors: <Color>[
                    Color(0xff1f005c),
                    Color(0xff5b0060),
                    Color(0xff870160),
                  ], // Gradient from https://learnui.design/tools/gradient-generator.html
                  tileMode: TileMode.mirror,
                ),
                borderRadius: BorderRadius.circular(10.0),
              ),
              width: MediaQuery.of(context).size.width * (3 / 8),
              height: MediaQuery.of(context).size.height / 5,
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    ElevatedButton.icon(
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Upload'),
                      onPressed: () async {
                        XFile galleryImg = await getImage();
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) =>
                                DisplayGalleryImage(displayImage: galleryImg),
                          ),
                        );
                        entry?.remove();
                      },
                    ),
                    ElevatedButton.icon(
                      icon: const Icon(Icons.photo_camera),
                      label: const Text("Capture"),
                      onPressed: () {
                        entry?.remove();
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
        ]),
      );
      final overlay = Overlay.of(context);
      overlay.insert(entry!);
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Sudoku Solver"),
        backgroundColor: Colors.deepPurple,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            child: Text("Camera will go here"),
            color: Colors.deepPurple,
          ),
          Text("Take a picture"),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        child: Icon(Icons.add),
        onPressed: () {
          showOverlayOptions();
          // log('--> OpenCV version ${nativeOpenCV.cvVersion()}');
          // Navigator.push(
          //     context,
          //     MaterialPageRoute(
          //         builder: (_) =>
          //             Text(nativeOpenCV.hello())));
          /* await for the available cameras to be returned then navigate */
          // await availableCameras().then((value) => Navigator.push(
          //       context,
          //       MaterialPageRoute(
          //           builder: (_) => TakePictureScreen(cameras: value)),
          //     ));
        },
      ),
    );
  }

  Future<XFile> getImage() async {
    final pickedFile = await picker.pickImage(
        source: ImageSource.gallery,
        requestFullMetadata: true,
        imageQuality: 100,
        maxHeight: 1000,
        maxWidth: 1000);
    XFile xfilePick = pickedFile!;
    return xfilePick!;
  }
}

class DisplayGalleryImage extends StatelessWidget {
  final XFile displayImage;
  const DisplayGalleryImage({super.key, required this.displayImage});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview Page')),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Image.file(File(displayImage.path), fit: BoxFit.cover, width: 250),
          const SizedBox(height: 24),
          Text(displayImage.name)
        ]),
      ),
    );
  }
}

class TakePictureScreen extends StatefulWidget {
  const TakePictureScreen({Key? key, required this.cameras}) : super(key: key);

  final List<CameraDescription>? cameras;

  @override
  State<TakePictureScreen> createState() => _TakePictureScreenState();
}

class _TakePictureScreenState extends State<TakePictureScreen> {
  late CameraController _cameraController;
  bool _isRearCameraSelected = true;

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    initCamera(widget.cameras![0]);
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => CameraState(),
      child: Scaffold(
          body: SafeArea(
        child: Stack(children: [
          (_cameraController.value.isInitialized)
              ? CameraPreview(_cameraController)
              : Container(
                  color: Colors.black,
                  child: const Center(child: CircularProgressIndicator())),
          Align(
              alignment: Alignment.bottomCenter,
              child: Container(
                height: MediaQuery.of(context).size.height * 0.20,
                decoration: const BoxDecoration(
                    borderRadius:
                        BorderRadius.vertical(top: Radius.circular(24)),
                    color: Colors.black),
                child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Expanded(
                          child: IconButton(
                        padding: EdgeInsets.zero,
                        iconSize: 30,
                        icon: Icon(
                            _isRearCameraSelected
                                ? CupertinoIcons.switch_camera
                                : CupertinoIcons.switch_camera_solid,
                            color: Colors.white),
                        onPressed: () {
                          setState(() =>
                              _isRearCameraSelected = !_isRearCameraSelected);
                          initCamera(
                              widget.cameras![_isRearCameraSelected ? 0 : 1]);
                        },
                      )),
                      Expanded(
                          child: IconButton(
                        onPressed: takePicture,
                        iconSize: 50,
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(),
                        icon: const Icon(Icons.circle, color: Colors.white),
                      )),
                      const Spacer(),
                    ]),
              )),
        ]),
      )),
    );
  }

  Future initCamera(CameraDescription cameraDescription) async {
    _cameraController =
        CameraController(cameraDescription, ResolutionPreset.high);
    try {
      await _cameraController.initialize().then((_) {
        if (!mounted) return;
        setState(() {});
      });
    } on CameraException catch (e) {
      debugPrint("camera error $e");
    }
  }

  Future takePicture() async {
    if (!_cameraController.value.isInitialized) {
      return null;
    }
    if (_cameraController.value.isTakingPicture) {
      return null;
    }
    try {
      await _cameraController.setFlashMode(FlashMode.off);
      XFile picture = await _cameraController.takePicture();
      Navigator.push(
          context,
          MaterialPageRoute(
              builder: (context) => PreviewPage(
                    picture: picture,
                  )));
    } on CameraException catch (e) {
      debugPrint('Error occured while taking picture: $e');
      return null;
    }
  }
}

class PreviewPage extends StatelessWidget {
  const PreviewPage({Key? key, required this.picture}) : super(key: key);

  final XFile picture;

  @override
  Widget build(BuildContext context) {
    /* this state can be used to set and retrive parameters from the camera state */
    var state = Provider.of<CameraState>(context);
    state.lastPicutureTaken = picture;

    return Scaffold(
      appBar: AppBar(title: const Text('Preview Page')),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Image.file(File(picture.path), fit: BoxFit.cover, width: 250),
          const SizedBox(height: 24),
          Text(picture.name)
        ]),
      ),
    );
  }
}
