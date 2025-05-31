import 'dart:async';
import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart' as path;
import 'package:tuple/tuple.dart';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

class TFLiteInterpreterState extends ChangeNotifier {

  // the path the the .tflite model
  // final String _pathNumberModel = "assets/NumberModel_v0.tflite";
  final String _pathNumberModel = "assets/test_number_model.tflite";

  // the tflite interpreter used to run inference
  late tfl.Interpreter interpreter;

  bool initialized = false; // true if nn model has been loaded in

  /// Description: This function takes a preprocessed image, partitions it into
  /// a 9x9 grid, and runs digit inference on each cell.
  ///
  /// **Parameters**:
  /// - warpedImg: The image to parse and run inference upon.
  /// **Returns**:
  /// - A list of 81 tuples where each tuple contains its parsed grid and its
  /// number prediction.
  Future<List<Tuple2<XFile, int>>> getPartitionedPredictions(XFile warpedImg) async {
    if (!initialized) {
      // the neural network has not been loaded in yet
      interpreter = await tfl.Interpreter.fromAsset(_pathNumberModel);
      initialized = true;
    }

    // will hold the final result of 81 sub-images
    List<int> result = [];

    // will resize the image such that after partitioning, each sub-image
    // will be 50x50 pixels
    XFile resizedWarpedImg = await resizeImage(warpedImg, 50 * 9, 50 * 9);

    // partition the images and store them as a list
    List<Tuple2<XFile, List<List<List<double>>>>> partionedImgCells = await partitionImage(resizedWarpedImg);

    // initialize the predicted labels
    // Map<int, List<double>> outputs = {};
    // for (int i = 0; i < 81; i++) {
    //   outputs[i] = List<double>.filled(10, 0);
    // }

    // run the tflite model to obtain the predictions
    // List<double> single_output = List<double>.filled(10, 0);
    List<List<List<double>>> outputs = List.generate(81, (_) => List.generate(1, (_) => List.filled(10, 0.0)));
    for (int i = 0; i < partionedImgCells.length; i++) {
      interpreter.run(partionedImgCells[i].item2.reshape([1, 50, 50, 1]), outputs[i]);
    }
    // TODO: It would be nice to understand why this method does not work.
    // interpreter.runForMultipleInputs([partionedImgCells], outputs);

    // iterate through each partitioned image
    for (int i = 0; i < partionedImgCells.length; i++) {
      // get the cell's 10 logit outputs
      List<double> currOutput = outputs[i][0];

      // we use the reduce method to get the maximum logit value
      double maxPrediction = currOutput
          .reduce((maxVal, element) => element > maxVal ? element : maxVal);

      // we get the index where the maximum logit output occurs
      int predictedLabel = currOutput.indexOf(maxPrediction);

      // append the predicted label
      result.add(predictedLabel);
    }
    log("prediction results: ${result}");

    // put the parsed images and their predicted labels in the desired format
    List<Tuple2<XFile, int>> final_result = List.generate(81,
        (i) => Tuple2(partionedImgCells[i].item1, result[i]));

    return final_result;
  }


  /// Description: This function takes an image and partitions them into an even
  /// 9x9 grid. The resulting images are flattened and returned as a list of 81
  /// images.
  ///
  /// **Parameters**:
  /// - imageFile: The warped image of the sudoku puzzle which we wish to partition
  ///
  /// **Returns**:
  /// - List of tuples where the first tuple element is the parsed image as an XFile (best
  ///   compatible for rendering in Flutter), and the second element is the decoded image (the
  ///   image as a list of bytes, best for running inference)
  Future<List<Tuple2<XFile, List<List<List<double>>>>>> partitionImage(XFile imageFile) async {
    // Read the image from the XFile
    Uint8List imageData = await imageFile.readAsBytes();

    // Decode the image using the 'image' package
    img.Image? image = img.decodeImage(imageData);

    if (image == null) {
      throw Exception("Failed to decode image");
    }

    // Calculate the size of each partition
    int partitionWidth = (image.width / 9).floor();
    int partitionHeight = (image.height / 9).floor();

    List<List<List<List<double>>>> partitions_in_bytes = List.generate(
      81,
          (_) => List.generate(
        50,
            (_) => List.generate(
          50,
              (_) => [0.0], // 1 channel
        ),
      ),
    );
    List<img.Image> image_partitions = [];

    // iterate through each parsed image
    for (int i = 0; i < 81; i++) {
      int row = i ~/ 9;
      int col = i % 9;
      // Extract the sub-image
      img.Image partition = img.copyCrop(
        image,
        col * partitionWidth,
        row * partitionHeight,
        partitionWidth,
        partitionHeight,
      );
      image_partitions.add(partition);

      // iterate through each pixel of the image to convert the pixels
      // to grayscale
      for (int y = 0; y < 50; y++) {
        for (int x = 0; x < 50; x++) {
          int pixel = partition.getPixel(x, y);
          int gray = img.getLuminance(pixel);
          partitions_in_bytes[i][y][x][0] = gray / 255;
        }
      }
    }

    // convert all the images to XFiles
    List<XFile> xfile_partitions = await imagesToXFiles(image_partitions);

    // pair all the XFiles with their data in bytes
    List<Tuple2<XFile, List<List<List<double>>>>> partitions = List.generate(81,
        (i) => Tuple2(xfile_partitions[i], partitions_in_bytes[i]));

    return partitions;
  }

  /// Description: This takes an image and reformats it to a Float32List
  ///
  /// **Parameters**:
  /// - image: Converts an image to grayscale.
  ///
  /// **Parameters**:
  /// - A float list of the decoded image in grayscale.
  Float32List imageToFloat32(img.Image image) {
    int width = image.width;
    int height= image.height;
    final input = Float32List(width * height);
    int index = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getPixel(x, y);
        int gray = img.getLuminance(pixel);
        input[index++] = gray / 255.0;
      }
    }

    return input;
  }

  /// Description: Converts a list of images to a list of XFiles
  ///
  /// **Parameters**:
  /// - images: List of images to convert to XFile format.
  ///
  /// **Returns**:
  /// - List of images in XFile format.
  Future<List<XFile>> imagesToXFiles(List<img.Image> images) async {
    List<XFile> xfileImages = [];

    final tempDir = await getTemporaryDirectory();

    for (int i = 0; i < images.length; i++) {
      // Encode image to PNG format (or use encodeJpg if preferred)
      final imageBytes = img.encodePng(images[i]);

      // Create a unique file name
      final filePath = path.join(tempDir.path, 'image_$i.png');

      // Write bytes to file
      final file = File(filePath);
      await file.writeAsBytes(imageBytes);

      // Wrap in XFile
      xfileImages.add(XFile(file.path));
    }

    return xfileImages;
  }

  /// Description: Takes an image and resizes it to the desired width and
  /// height.
  ///
  /// **Parameters**:
  /// - xfile: The image to resize
  /// - width: Target width
  /// - height: Target height
  ///
  /// **Returns**:
  /// - The resized image.
  Future<XFile> resizeImage(XFile xfile, int width, int height) async {
    final bytes = await xfile.readAsBytes();
    final image = img.decodeImage(bytes);

    if (image == null) {
      throw Exception("Failed to decode image");
    }

    final resizedImage = img.copyResize(image, width: width, height: height);

    final directory = await getTemporaryDirectory();
    final resizedImagePath = '${directory.path}/resized_image.png';

    final resizedImageFile =
        await File(resizedImagePath).writeAsBytes(img.encodePng(resizedImage));

    return XFile(resizedImageFile.path);
  }
}
