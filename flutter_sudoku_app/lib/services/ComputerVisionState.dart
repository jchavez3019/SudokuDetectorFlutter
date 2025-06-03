import "dart:io";
import "dart:typed_data";

import "package:camera/camera.dart";
import "package:flutter/material.dart";
// import "package:flutter_sudoku_app/detector/sudoku_detector.dart";
import "package:flutter_sudoku_app/services/TFLiteNeuralNetwork.dart";
import "package:native_opencv/native_opencv.dart";
import "package:path_provider/path_provider.dart";
import "package:provider/provider.dart";
import "package:tuple/tuple.dart";

class ComputerVisionState extends ChangeNotifier {

  // NativeOpenCV class that allows us to call our custom C++ code for
  // image processing.
  final nativeOpenCV = NativeOpencv();

  // This class provides methods that allow us to parse and run inference
  // on an image of a Sudoku board.
  final tfliteNN = TFLiteNeuralNetwork();

  /* private attributes */

  // attributes related to image processing
  XFile? originalImage;
  XFile? warpedImage;
  bool? isImageUploaded;
  bool errorRunningInference = false;
  String ?inferenceErrorMsg;

  // attributes related to inference
  List<Tuple2<XFile, int>>? cachedPredictions; // will hold the initial partitioned cells and their predictions
  bool loadedPredictions = false; // true once inference has been run on the processed image
  List<int>? modifiedPredictions; // holds the initial predictions but can be modified by the user (e.g., maybe some were incorrect)
  int num_edited = 0; // number of labels that have been editted by the user
  double accuracy = 0; // accuracy of the model thus far

  /* notifiers */

  updateImage(XFile newImage, bool isUpload) {
    originalImage = newImage;
    isImageUploaded = isUpload;
    warpedImage = null;
    loadedPredictions = false;
    cachedPredictions = null;
    modifiedPredictions = null;
    errorRunningInference = false;
    inferenceErrorMsg = null;
    getWarpedImage();
    notifyListeners();
  }

  /// Description:  Given an image, uses the `detectSudokuPuzzle` method to
  ///               process it. The processed image is returned in XFile format.
  ///
  /// **Parameters**:
  /// - `displayImage`: The image to process
  ///
  /// **Returns**:
  /// - The final processed image, encoded as an XFile for rendering.
  getWarpedImage() async {
    Uint8List bytes = await originalImage!.readAsBytes();
    File tempFile = await saveImageToFile(
        nativeOpenCV.detectSudokuPuzzle(bytes),
        prefix: "warpedImage");
    warpedImage =  XFile(tempFile.path);
    notifyListeners();
  }

  /// Description:  We partition the processed image into 9x9 cells/images,
  ///               and then we run inference on this batch of cells. Once
  ///               inference is complete, we update the internal state.
  runInference() {
    tfliteNN.getPartitionedPredictions(warpedImage!).then((ret) {
      cachedPredictions = ret;
      modifiedPredictions = ret.map((e) => e.item2).toList();
      loadedPredictions = true;
      notifyListeners();
    }).catchError((error) {
      loadedPredictions = true;
      errorRunningInference = true;
      inferenceErrorMsg = error;
      debugPrint('Inference error: $inferenceErrorMsg');
      notifyListeners();
    });
  }

  updateModifiedPredictions(int idx, int val) {
    modifiedPredictions![idx] = val;

    var new_num_editted = 0;
    var new_num_correct = 0;
    for (int i = 0; i < 81; i++) {
      // if (index != i && cachedPredictions[i].item2 != modifiedPredictions![i] ||
      //     (index == i && cachedPredictions[i].item2 != val)) {
      if (cachedPredictions![i].item2 != modifiedPredictions![i]) {
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
    num_edited = new_num_editted;
    accuracy = new_accuracy;
    notifyListeners();
  }

  /// Description: Runs the C++ code for extracting a warped image which will
  ///              will contained a bird's eye view of the sudoku board.
  ///
  /// **Parameters**:
  /// - `originalImage`: The image to process
  ///
  /// **Returns**:
  /// - The processed image, decoded as a list of bytes.
  Uint8List detectSudokuPuzzle(Uint8List originalImage) =>
      nativeOpenCV.detectSudokuPuzzle(originalImage);


  /// Description:  A helper function that allows us to encode a list of bytes
  ///               into a .jpg image. The image is stored in a temporary file
  ///               and the user should not expect it to be saved permanently.
  ///
  /// **Parameters**:
  /// - `data`: A list of bytes corresponding to a decoded image.
  ///
  /// **Returns**:
  /// - A file path to where the encoded image is temporarily stored.
  Future<File> saveImageToFile(Uint8List data,
      {String prefix = "temp_image"}) async {
    Directory tempDir = await getTemporaryDirectory();
    File tempFile = File(
        '${tempDir.path}/${prefix}_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await tempFile.writeAsBytes(data);
    return tempFile;
  }


}
