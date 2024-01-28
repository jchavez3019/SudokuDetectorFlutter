// ignore_for_file: camel_case_types

import 'dart:async';
import 'dart:developer';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';

// Load our C lib
final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open("libnative_opencv.so")
    : DynamicLibrary.process();

// C Function Signatures
typedef _c_version = Pointer<Utf8> Function();
typedef _c_hello = Pointer<Utf8> Function();
typedef _c_initDetector = Void Function(
    Pointer<Uint8> markerPngBytes, Int32 inSize, Int32 bits);
typedef _c_destroyDetector = Void Function();
typedef _c_detect = Pointer<Float> Function(Int32 width, Int32 height,
    Int32 rotation, Pointer<Uint8> bytes, Bool isYUV, Pointer<Int32> outCount);
typedef _c_rotateImage = Pointer<Uint8> Function(
    Pointer<Uint8> originalImage, Int32 inSize, Pointer<Int32> finalSize);
typedef _c_detectSudokuPuzzle = Pointer<Uint8> Function(
    Pointer<Uint8> originalImage, Int32 inSize, Pointer<Int32> finalSize);

// Dart Function Signatures
typedef _dart_version = Pointer<Utf8> Function();
typedef _dart_hello = Pointer<Utf8> Function();
typedef _dart_initDetector = void Function(
    Pointer<Uint8> markerPngBytes, int inSize, int bits);
typedef _dart_destroyDetector = void Function();
typedef _dart_detect = Pointer<Float> Function(int width, int height,
    int rotation, Pointer<Uint8> bytes, bool isYUV, Pointer<Int32> outCount);
typedef _dart_rotateImage = Pointer<Uint8> Function(
    Pointer<Uint8> originalImage, int inSize, Pointer<Int32> finalSize);
typedef _dart_detectSudokuPuzzle = Pointer<Uint8> Function(
    Pointer<Uint8> originalImage, int inSize, Pointer<Int32> finalSize);

// Create dart functions that invoke the C function
final _version = nativeLib.lookupFunction<_c_version, _dart_version>("version");
final _initDetector = nativeLib
    .lookupFunction<_c_initDetector, _dart_initDetector>("initDetector");
final _destroyDetector =
    nativeLib.lookupFunction<_c_destroyDetector, _dart_destroyDetector>(
        "destroyDetector");
final _detect = nativeLib.lookupFunction<_c_detect, _dart_detect>("detect");
final _rotateImage =
    nativeLib.lookupFunction<_c_rotateImage, _dart_rotateImage>("rotateImage");
final _detectSudokuPuzzle =
    nativeLib.lookupFunction<_c_detectSudokuPuzzle, _dart_detectSudokuPuzzle>(
        "detectSudokuPuzzle");

final _hello = nativeLib.lookupFunction<_c_hello, _dart_hello>("hello");

class NativeOpencv {
  static const MethodChannel _channel = MethodChannel("native_opencv");
  Pointer<Uint8>? _imageBuffer;

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod("getPlatformVersion");
    return version;
  }

  String cvVersion() {
    return _version().toDartString();
  }

  void initDetector(Uint8List markerPngBytes, int bits) {
    var totalSize = markerPngBytes.lengthInBytes;
    var imgBuffer = malloc.allocate<Uint8>(totalSize);

    /* typing as list allows us to use the .setAll method */
    Uint8List bytes = imgBuffer.asTypedList(totalSize);
    /* copies over all the bytes from markPngBytes into imgBuffer */
    bytes.setAll(0, markerPngBytes);
    log('--> Before Started _initDetector from NativeOpenCV class');

    _initDetector(imgBuffer, totalSize, bits);

    log('--> Started _initDetector from NativeOpenCV class');

    malloc.free(imgBuffer);
  }

  Uint8List rotateImage(Uint8List originalImage) {
    var totalSize = originalImage.lengthInBytes;
    var imgBuffer = malloc.allocate<Uint8>(totalSize);
    Uint8List bytes = imgBuffer.asTypedList(totalSize);
    bytes.setAll(0, originalImage);

    Pointer<Int32> finalSizePtr = malloc.allocate<Int32>(1);

    var rotatedImage = _rotateImage(imgBuffer, totalSize, finalSizePtr);

    var finalSize = finalSizePtr.value;

    malloc.free(imgBuffer);
    malloc.free(finalSizePtr);

    return rotatedImage.asTypedList(finalSize);
  }

  Uint8List detectSudokuPuzzle(Uint8List originalImage) {
    var totalSize = originalImage.lengthInBytes;
    var imgBuffer = malloc.allocate<Uint8>(totalSize);
    Uint8List bytes = imgBuffer.asTypedList(totalSize);
    bytes.setAll(0, originalImage);

    Pointer<Int32> finalSizePtr = malloc.allocate<Int32>(1);

    var sudokuImage = _detectSudokuPuzzle(imgBuffer, totalSize, finalSizePtr);

    var finalSize = finalSizePtr.value;

    malloc.free(imgBuffer);
    malloc.free(finalSizePtr);

    return sudokuImage.asTypedList(finalSize);
  }

  void destroy() {
    _destroyDetector();
    if (_imageBuffer != null) {
      malloc.free(_imageBuffer!);
    }
  }

  Float32List detect(int width, int height, int rotation, Uint8List yBuffer,
      Uint8List? uBuffer, Uint8List? vBuffer) {
    var ySize = yBuffer.lengthInBytes;
    var uSize = uBuffer?.lengthInBytes ?? 0;
    var vSize = vBuffer?.lengthInBytes ?? 0;
    var totalSize = ySize + uSize + vSize;

    /* allocate if it is null */
    _imageBuffer ??= malloc.allocate<Uint8>(totalSize);

    /* we always have at least 1 plane, on Android it is the yPlane. On iOS, it's the rgba plane */
    Uint8List _bytes = _imageBuffer!.asTypedList(totalSize);
    _bytes.setAll(0, yBuffer);

    if (Platform.isAndroid) {
      // swap u and v buffer for opencv
      _bytes.setAll(ySize, vBuffer!);
      _bytes.setAll(ySize + vSize, uBuffer!);
    }

    Pointer<Int32> outCount = malloc.allocate<Int32>(1);
    var res = _detect(width, height, rotation, _imageBuffer!,
        Platform.isAndroid ? true : false, outCount);

    final count = outCount.value;

    malloc.free(outCount);
    return res.asTypedList(count);
  }

  String hello() {
    return _hello().toDartString();
  }
}
