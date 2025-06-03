import 'dart:developer';
import 'dart:typed_data';
import 'package:tuple/tuple.dart';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sudoku_app/services/ComputerVisionState.dart';
import 'package:flutter_sudoku_app/services/TFLiteNeuralNetwork.dart';
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
    // We enable the  ComputerVisionState provider
    // so that we can keep track of the current image,
    // processed image, predictions, etc. It also gives us
    // access to computer vision methods such as our backend C++ code
    // as well as access to our trained NN for running inference.
    return ChangeNotifierProvider<ComputerVisionState>(
        create: (context) => ComputerVisionState(),
        child: MaterialApp(
          title: 'Flutter Demo',
          theme: appTheme,
          home: MyHomePage(title: 'Flutter Demo Home Page'),
        ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final String title;

  MyHomePage({super.key, required this.title});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  OverlayEntry? entry;
  XFile? chosenGalleryImage;


  @override
  Widget build(BuildContext context) {

    // disable listen since we don't want to rebuild this widget whenever
    // notifyListeners is invoked
    ComputerVisionState cvState = Provider.of<ComputerVisionState>(context, listen: false);

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
          ), // removes the overlay if the screen outside is touched
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
              // we ensure the width is 3/8 of the width and 1/5 of the height
              // of the screen
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

                          // record that this is the new image we are working with
                          cvState.updateImage(galleryImg, false);

                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) =>
                              DisplayPreviewImage(),
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
                        getImage(true).then((XFile galleryImg) {

                          // record that this is the image we are working with
                          cvState.updateImage(galleryImg, true);

                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) =>
                              DisplayPreviewImage(),
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

class DisplayPreviewImage extends StatelessWidget {
  /*
  This class displays a preview image of a warped version of the user's selected
  photo. They can then choose to proceed if the warped image looks correct
  or they can reattempt to upload an image. 
  */

  DisplayPreviewImage({super.key});

  @override
  Widget build(BuildContext context) {

    // disable listen since we don't want to rebuild this widget whenever
    // notifyListeners is invoked
    ComputerVisionState cvState = Provider.of<ComputerVisionState>(context, listen: false);
    final bool isUpload = cvState.isImageUploaded!;

    // we only care about when the warpedImage gets updated in the provider
    final warpedImage = context.select<ComputerVisionState, XFile?>(
          (state) => state.warpedImage,
    );

    if (warpedImage == null) {
      return const Loader();
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Preview Page'),
        backgroundColor: Colors.deepPurple,
      ),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
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
                      getImage(isUpload).then((XFile newDisplayImg) {
                        cvState.updateImage(newDisplayImg, isUpload);
                      });
                    },
                    icon: const Icon(Icons.cancel),
                    label: const Text("Retry")),
                ElevatedButton.icon(
                    onPressed: () {
                      // the user has confirmed to proceed with the processed
                      // image, let's partition it and run inference on each
                      // cell
                      cvState.runInference();

                      // proceed to the next page where we display
                      // the predicted labels
                      Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (BuildContext context) =>
                                DisplayInferredSudokuPuzzle(),
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
}

/// Description:  Returns an image either from the user's gallery or a live
///               picture using their camera.
/// **Paramaters**:
///
/// - `useCamera`: flag set when the live camera is to be used, else use the gallery
///
/// **Returns**:
///
/// - xfilePick: The chosen image in XFile format
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
  DisplayInferredSudokuPuzzle({super.key});

  @override
  State<DisplayInferredSudokuPuzzle> createState() =>
      _DisplayInferredSudokuPuzzleState();
}

class _DisplayInferredSudokuPuzzleState
    extends State<DisplayInferredSudokuPuzzle> {
  /* private variables */
  bool _show_labels = true; // when true, overlays the 9x9 grid of images with their labels

  @override
  Widget build(BuildContext context) {

    // get the provider and listen for changes
    final cvState = Provider.of<ComputerVisionState>(context);

    // flag set to true once the model has made its final predictions
    bool loadedPredictions = cvState.loadedPredictions;

    // the predictions have not been made yet, return the loader
    if (!loadedPredictions) {
      return const Scaffold(body: Loader());
    }

    // predictions have been made, we can get the initial results
    String? errorMsg = cvState.inferenceErrorMsg;

    if (errorMsg != null) {
      return Scaffold(
        body: Center(
          child: Text(
            errorMsg,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
      );
    }

    // reaching this point means that inference was successfull and we can
    // unpack the partitioned cells, initial predictions, and our copy
    // of the predictions
    final List<Tuple2<XFile, int>> cachedPredictions = cvState.cachedPredictions!;
    List<int> _modified_labels = cvState.modifiedPredictions!;
    int _num_edited = cvState.num_edited;
    double _accuracy = cvState.accuracy;

    //
    final List<XFile> image_cells = cachedPredictions.map((e) => e.item1).toList();
    final labels = _modified_labels;

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

                    // give the user a number pad to update the label of a cell
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
                      // The user has updated a label, e.g., its prediction
                      // was incorrect. Update the predictions in the provider,
                      // this will then trigger a widget rebuild.
                      cvState.updateModifiedPredictions(index, newLabel);
                    }
                  },

                  // This stack will show each cell's image. If _show_labels is
                  // true, we will overlay the images with their predicted labels.
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
