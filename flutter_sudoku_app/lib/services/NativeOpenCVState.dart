import "dart:io";
import "dart:typed_data";

import "package:camera/camera.dart";
import "package:flutter/material.dart";
import "package:native_opencv/native_opencv.dart";
import "package:path_provider/path_provider.dart";
import "package:provider/provider.dart";

class NativeOpencvState extends ChangeNotifier {
  final nativeOpenCV = NativeOpencv();

  /// Description: Runs the C++ code for extracting a warped image which will
  ///              will contained a bird's eye view of the sudoku board.
  ///
  /// **Parameters**:
  /// - originalImage: The image to process
  ///
  /// **Returns**:
  /// - The processed image, decoded as a list of bytes.
  Uint8List detectSudokuPuzzle(Uint8List originalImage) =>
      nativeOpenCV.detectSudokuPuzzle(originalImage);

  /// Description:  Given an image, uses the `detectSudokuPuzzle` method to
  ///               process it. The processed image is returned in XFile format.
  ///
  /// **Parameters**:
  /// - displayImage: The image to process
  ///
  /// **Returns**:
  /// - The final processed image, encoded as an XFile for rendering.
  Future<XFile> getWarpedImage(XFile displayImage) async {
    Uint8List bytes = await displayImage.readAsBytes();
    File tempFile = await saveImageToFile(
        nativeOpenCV.detectSudokuPuzzle(bytes),
        prefix: "warpedImage");
    return XFile(tempFile.path);
  }

  /// Description:  A helper function that allows us to encode a list of bytes
  ///               into a .jpg image. The image is stored in a temporary file
  ///               and the user should not expect it to be saved permanently.
  ///
  /// **Parameters**:
  /// - data: A list of bytes corresponding to a decoded image.
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
