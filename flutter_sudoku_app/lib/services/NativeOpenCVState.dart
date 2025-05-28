import "dart:io";
import "dart:typed_data";

import "package:camera/camera.dart";
import "package:flutter/material.dart";
import "package:native_opencv/native_opencv.dart";
import "package:path_provider/path_provider.dart";
import "package:provider/provider.dart";

class NativeOpencvState extends ChangeNotifier {
  final nativeOpenCV = NativeOpencv();

  Uint8List detectSudokuPuzzle(Uint8List originalImage) =>
      nativeOpenCV.detectSudokuPuzzle(originalImage);

  Future<XFile> getWarpedImage(XFile displayImage) async {
    Uint8List bytes = await displayImage.readAsBytes();
    File tempFile = await saveImageToFile(
        nativeOpenCV.detectSudokuPuzzle(bytes),
        prefix: "warpedImage");
    return XFile(tempFile.path);
  }

  Future<File> saveImageToFile(Uint8List data,
      {String prefix = "temp_image"}) async {
    Directory tempDir = await getTemporaryDirectory();
    File tempFile = File(
        '${tempDir.path}/${prefix}_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await tempFile.writeAsBytes(data);
    return tempFile;
  }
}
