import 'dart:developer';
import 'dart:typed_data';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_sudoku_app/services/loading.dart';
import 'package:flutter_sudoku_app/theme.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_sudoku_app/camera_state.dart';
import 'package:native_opencv/native_opencv.dart';
import 'package:path_provider/path_provider.dart';
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
      theme: appTheme,
      home: MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final String title;
  bool navigateFurther = false;

  MyHomePage({super.key, required this.title});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool get navigateFurther => widget.navigateFurther;
  set navigateFurther(bool newValue) {
    setState(() {
      widget.navigateFurther = newValue;
    });
  }

  OverlayEntry? entry;
  XFile? chosenGalleryImage;

  late File selectedImage;

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
              entry?.dispose();
              entry = null;
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
                      onPressed: () {
                        getImage(false).then((XFile galleryImg) {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) =>
                                  // DisplayGalleryImage(displayImage: galleryImg),
                                  DisplayPreviewImage(
                                displayImage: galleryImg,
                                isUpload: true,
                              ),
                            ),
                          );
                          entry?.remove();
                          entry?.dispose();
                          entry = null;
                        });
                      },
                    ),
                    ElevatedButton.icon(
                      icon: const Icon(Icons.photo_camera),
                      label: const Text("Capture"),
                      onPressed: () {
                        getImage(false).then((XFile galleryImg) {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) =>
                                  // DisplayGalleryImage(displayImage: galleryImg),
                                  DisplayPreviewImage(
                                displayImage: galleryImg,
                                isUpload: true,
                              ),
                            ),
                          );
                          entry?.remove();
                          entry?.dispose();
                          entry = null;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
          ),
        ]),
      );
      Overlay.of(context).insert(entry!);
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Sudoku Solver"),
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text("Welcome to the Sudoku Solver",
                style: Theme.of(context).textTheme.bodyLarge),
            Text(
              "This app is capable of recognizing Sudoku puzzles from images and solving them. To try this out, click on the 'plus' button on the bottom right and either upload or capture an image. Enjoy!",
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        child: const Icon(Icons.add),
        onPressed: () {
          showOverlayOptions();
        },
      ),
    );
  }
}

class DisplayPreviewImage extends StatefulWidget {
  XFile displayImage;
  final bool isUpload;

  DisplayPreviewImage(
      {super.key, required this.displayImage, required this.isUpload});

  @override
  State<DisplayPreviewImage> createState() => _DisplayPreviewImageState();
}

class _DisplayPreviewImageState extends State<DisplayPreviewImage> {
  /* getters */
  XFile get displayImage => widget.displayImage;
  bool get isUpload => widget.isUpload;

  /* setters */
  set displayImage(XFile newImage) {
    setState(() {
      widget.displayImage = newImage;
    });
  }

  final nativeOpenCV = NativeOpencv();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
        future: _getWarpedImage(),
        builder: (context, snapshot) {
          /* the snapshot has not arrived or there was an error */
          if (!snapshot.hasData || snapshot.hasError) {
            return const Loader();
          } else {
            XFile warpedImage = snapshot.data!;

            return Scaffold(
              appBar: AppBar(
                title: const Text('Preview Page'),
                backgroundColor: Colors.deepPurple,
              ),
              body: Center(
                child: Column(mainAxisSize: MainAxisSize.min, children: [
                  // Image.file(File(displayImage.path), fit: BoxFit.cover, width: 250),
                  Image.file(File(warpedImage.path),
                      fit: BoxFit.cover, width: 250),
                  const SizedBox(height: 24),
                  Text(warpedImage.name),
                  Padding(
                    padding: const EdgeInsets.only(top: 20.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                            onPressed: () {
                              getImage(!isUpload).then((XFile newDisplayImg) {
                                displayImage = newDisplayImg;
                              });
                            },
                            icon: const Icon(Icons.cancel),
                            label: const Text("Retry")),
                        ElevatedButton.icon(
                            onPressed: () {},
                            icon: const Icon(Icons.check_circle),
                            label: const Text("Proceed")),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Text(
                      "If the Sudoku Puzzle doesn't look like it was correctly extracted, you can try to upload/capture it again. Otherwise proceed and we'll get to solving!",
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                ]),
              ),
            );
          }
        });
  }

  Future<XFile> _getWarpedImage() async {
    Uint8List bytes = await widget.displayImage.readAsBytes();
    File tempFile = await _saveImageToFile(
        nativeOpenCV.detectSudokuPuzzle(bytes),
        prefix: "warpedImage");
    return XFile(tempFile.path);
  }

  Future<File> _saveImageToFile(Uint8List data,
      {String prefix = "temp_image"}) async {
    Directory tempDir = await getTemporaryDirectory();
    File tempFile = File(
        '${tempDir.path}/${prefix}_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await tempFile.writeAsBytes(data);
    return tempFile;
  }
}

Future<XFile> getImage(bool useCamera) async {
  final pickedFile = await ImagePicker().pickImage(
      source: useCamera ? ImageSource.camera : ImageSource.gallery,
      requestFullMetadata: true,
      imageQuality: 100,
      maxHeight: 1000,
      maxWidth: 1000);
  XFile xfilePick = pickedFile!;
  return xfilePick;
}


/* everything below is a real time camera stream that is not needed
but can be useful for some future project */


// class TakePictureScreen extends StatefulWidget {
//   const TakePictureScreen({Key? key, required this.cameras}) : super(key: key);

//   final List<CameraDescription>? cameras;

//   @override
//   State<TakePictureScreen> createState() => _TakePictureScreenState();
// }

// class _TakePictureScreenState extends State<TakePictureScreen> {
//   late CameraController _cameraController;
//   bool _isRearCameraSelected = true;

//   @override
//   void dispose() {
//     _cameraController.dispose();
//     super.dispose();
//   }

//   @override
//   void initState() {
//     super.initState();
//     initCamera(widget.cameras![0]);
//   }

//   @override
//   Widget build(BuildContext context) {
//     return ChangeNotifierProvider(
//       create: (_) => CameraState(),
//       child: Scaffold(
//           body: SafeArea(
//         child: Stack(children: [
//           (_cameraController.value.isInitialized)
//               ? CameraPreview(_cameraController)
//               : Container(
//                   color: Colors.black,
//                   child: const Center(child: CircularProgressIndicator())),
//           Align(
//               alignment: Alignment.bottomCenter,
//               child: Container(
//                 height: MediaQuery.of(context).size.height * 0.20,
//                 decoration: const BoxDecoration(
//                     borderRadius:
//                         BorderRadius.vertical(top: Radius.circular(24)),
//                     color: Colors.black),
//                 child: Row(
//                     crossAxisAlignment: CrossAxisAlignment.center,
//                     children: [
//                       Expanded(
//                           child: IconButton(
//                         padding: EdgeInsets.zero,
//                         iconSize: 30,
//                         icon: Icon(
//                             _isRearCameraSelected
//                                 ? CupertinoIcons.switch_camera
//                                 : CupertinoIcons.switch_camera_solid,
//                             color: Colors.white),
//                         onPressed: () {
//                           setState(() =>
//                               _isRearCameraSelected = !_isRearCameraSelected);
//                           initCamera(
//                               widget.cameras![_isRearCameraSelected ? 0 : 1]);
//                         },
//                       )),
//                       Expanded(
//                           child: IconButton(
//                         onPressed: takePicture,
//                         iconSize: 50,
//                         padding: EdgeInsets.zero,
//                         constraints: const BoxConstraints(),
//                         icon: const Icon(Icons.circle, color: Colors.white),
//                       )),
//                       const Spacer(),
//                     ]),
//               )),
//         ]),
//       )),
//     );
//   }

//   Future initCamera(CameraDescription cameraDescription) async {
//     _cameraController =
//         CameraController(cameraDescription, ResolutionPreset.high);
//     try {
//       await _cameraController.initialize().then((_) {
//         if (!mounted) return;
//         setState(() {});
//       });
//     } on CameraException catch (e) {
//       debugPrint("camera error $e");
//     }
//   }

//   Future takePicture() async {
//     if (!_cameraController.value.isInitialized) {
//       return null;
//     }
//     if (_cameraController.value.isTakingPicture) {
//       return null;
//     }
//     try {
//       await _cameraController.setFlashMode(FlashMode.off);
//       XFile picture = await _cameraController.takePicture();
//       Navigator.push(
//           context,
//           MaterialPageRoute(
//               builder: (context) => PreviewPage(
//                     picture: picture,
//                   )));
//     } on CameraException catch (e) {
//       debugPrint('Error occured while taking picture: $e');
//       return null;
//     }
//   }
// }

// class PreviewPage extends StatelessWidget {
//   const PreviewPage({Key? key, required this.picture}) : super(key: key);

//   final XFile picture;

//   @override
//   Widget build(BuildContext context) {
//     /* this state can be used to set and retrive parameters from the camera state */
//     var state = Provider.of<CameraState>(context);
//     state.lastPicutureTaken = picture;

//     return Scaffold(
//       appBar: AppBar(title: const Text('Preview Page')),
//       body: Center(
//         child: Column(mainAxisSize: MainAxisSize.min, children: [
//           Image.file(File(picture.path), fit: BoxFit.cover, width: 250),
//           const SizedBox(height: 24),
//           Text(picture.name)
//         ]),
//       ),
//     );
//   }
// }
