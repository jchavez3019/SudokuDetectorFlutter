// ignore_for_file: camel_case_types

import 'dart:async';
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

// Dart Function Signatures
typedef _dart_version = Pointer<Utf8> Function();
typedef _dart_hello = Pointer<Utf8> Function();

// Create dart functions that invoke the C function
final _version = nativeLib.lookupFunction<_c_version, _dart_version>('version');

final _hello = nativeLib.lookupFunction<_c_hello, _dart_hello>("hello");

class NativeOpencv {
  static const MethodChannel _channel = MethodChannel("native_opencv");

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod("getPlatformVersion");
    return version;
  }

  String cvVersion() {
    return _version().toDartString();
  }

  String hello() {
    return _hello().toDartString();
  }
}
