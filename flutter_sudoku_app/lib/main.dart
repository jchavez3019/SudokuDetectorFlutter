import 'dart:developer';
import 'dart:typed_data';
import 'package:tuple/tuple.dart';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sudoku_app/services/NativeOpenCVState.dart';
import 'package:flutter_sudoku_app/services/TFLiteInterpreterState.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_sudoku_app/services/loading.dart';
import 'package:flutter_sudoku_app/theme.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_sudoku_app/camera_state.dart';
import 'package:native_opencv/native_opencv.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

// import 'package:pytorch_mobile/pytorch_mobile.dart';
// import 'package:pytorch_mobile/model.dart';
// import 'package:flutter_pytorch/flutter_pytorch.dart';
// import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

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
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (BuildContext context) =>
                                  ChangeNotifierProvider<NativeOpencvState>(
                                create: (_) => NativeOpencvState(),
                                child: DisplayPreviewImage(
                                  displayImage: galleryImg,
                                  isUpload: true,
                                ),
                              ),
                              // DisplayGalleryImage(displayImage: galleryImg),
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
  /*
  This class displays a preview image of a warped version of the user's selected
  photo. They can then choose to proceed if the warped image looks correct
  or they can reattempt to upload an image. 
  */
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

  @override
  Widget build(BuildContext context) {
    NativeOpencvState nativeOpenCV = Provider.of<NativeOpencvState>(context);

    return FutureBuilder<XFile>(
        future: nativeOpenCV.getWarpedImage(displayImage),
        builder: (context, snapshot) {
          /* the snapshot has not arrived or there was an error */
          if (!snapshot.hasData || snapshot.hasError) {
            return const Loader();
          } else if (snapshot.hasError) {
            return ErrorWidget(Text(
              "Error getting the warped image",
              style: Theme.of(context).textTheme.bodyMedium,
            ));
          } else {
            // we have recieved the warpedImage successfully

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
                            onPressed: () {
                              Navigator.of(context).push(MaterialPageRoute(
                                builder: (BuildContext context) =>
                                    ChangeNotifierProvider<
                                            TFLiteInterpreterState>(
                                        create: (_) => TFLiteInterpreterState(),
                                        child: DisplayInferredSudokuPuzzle(
                                            warpedImage: warpedImage)),
                              ));
                            },
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

class DisplayInferredSudokuPuzzle extends StatefulWidget {
  /* 
  This class takes the previously warped image and infers what digit is in 
  each cell. 
  */
  XFile warpedImage;

  DisplayInferredSudokuPuzzle({super.key, required this.warpedImage});

  @override
  State<DisplayInferredSudokuPuzzle> createState() =>
      _DisplayInferredSudokuPuzzleState();
}

class _DisplayInferredSudokuPuzzleState
    extends State<DisplayInferredSudokuPuzzle> {
  /* private variables */
  bool _show_labels = true; // when true, overlays the 9x9 grid of images with their labels
  List<Tuple2<XFile, int>>? _cached_predictions; // will hold the initial partitioned cells and their predictions
  List<int>? _modified_labels; // holds the initial predictions but can be modified by the user (e.g., maybe some were incorrect)
  bool _is_loading = true; // false once the initial predictions have been made
  String? _error; // non-empty if an error encountered when running predictions

  int _num_correct = 0; // number of label predictions that are correct
  int _num_edited = 0; // number of labels that have been editted by the user
  double _accuracy = 0; // accuracy of the model thus far

  /* getters */
  XFile get warpedImage => widget.warpedImage;
  /* setters */
  set warpedImage(XFile newImage) {
    setState(() {});
    widget.warpedImage = newImage;
  }

  @override
  void initState() {
    super.initState();
    _loadPredictions();
  }

  Future<void> _loadPredictions() async {
    try {
      // partition the warped image and run inference on each cell
      final tflis = Provider.of<TFLiteInterpreterState>(context, listen: false);
      final result = await tflis.getPartitionedPredictions(warpedImage);
      setState(() {
        _cached_predictions = result;
        _modified_labels = result.map((e) => e.item2).toList();
        _is_loading = false;
        _num_correct = 81;
        _accuracy = 100.0;
      });
    } catch (e) {
      // there was an error trying to run inference
      setState(() {
        _error = "Error getting the inferred image.";
        _is_loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_is_loading) {
      return const Scaffold(body: Loader());
    }

    if (_error != null || _modified_labels == null) {
      return Scaffold(
        body: Center(
          child: Text(
            _error ?? "Unknown error",
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
      );
    }

    final image_cells = _cached_predictions!.map((e) => e.item1).toList();
    final labels = _modified_labels!;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Partitioned Predictions'),
        actions: [
          IconButton(
              icon: Icon(_show_labels ? Icons.visibility_off : Icons.visibility),
              onPressed: () {
                setState(() {
                  _show_labels = !_show_labels;
                });
              }
          )
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 9,
                crossAxisSpacing: 2.0,
                mainAxisSpacing: 2.0,
              ),
              itemCount: image_cells.length,
              itemBuilder: (context, index) {
                final image = image_cells[index];
                final label = labels[index];

                return GestureDetector(
                  onTap: () async {
                    final controller = TextEditingController(text: '$label');
                    int? newLabel = await showDialog<int>(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: const Text("Edit Label"),
                        content: TextField(
                          controller: controller,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(labelText: "New label"),
                        ),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.of(context).pop(),
                            child: const Text("Cancel"),
                          ),
                          TextButton(
                            onPressed: () {
                              final parsed = int.tryParse(controller.text);
                              if (parsed != null) {
                                Navigator.of(context).pop(parsed);
                              }
                            },
                            child: const Text("OK"),
                          ),
                        ],
                      ),
                    );

                    if (newLabel != null) {
                      var new_num_editted = 0;
                      var new_num_correct = 0;
                      for (int i = 0; i < 81; i++) {
                        if (index != i && _cached_predictions![i].item2 != _modified_labels![i] ||
                            (index == i && _cached_predictions![i].item2 != newLabel)) {
                          // count the number of modified labels that do not match the
                          // originally predicted labels
                          new_num_editted++;
                        }
                        else {
                          // if the modified label matches the original prediction,
                          // we assume that this means the prediction was correct
                          new_num_correct++;
                        }
                      }
                      var new_accuracy = (new_num_correct / 81.0) * 100.0;
                      setState(() {
                        _modified_labels![index] = newLabel;
                        _num_edited = new_num_editted;
                        _num_correct = new_num_correct;
                        _accuracy = new_accuracy;
                      });
                    }
                  },
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      Image.file(File(image.path), fit: BoxFit.cover),
                      if (_show_labels)
                        Positioned(
                          bottom: 10,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                            color: Colors.black54,
                            child: Text(
                              "$label",
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                );
              },
            ),
          ),
          // instruction block for the user
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Text(
              "Tap on any cell to edit its label. Use the toggle in the top-right to show/hide labels.",
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          // display the accuracy of the inference
          Padding(
          padding: const EdgeInsets.only(bottom: 16.0),
          child: Text(
          "Edited: $_num_edited / 81 | Accuracy: ${_accuracy.toStringAsFixed(1)}%",
          style: Theme.of(context).textTheme.bodyMedium,
          ),
          ),
        ],
      ),
    );

  }
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
