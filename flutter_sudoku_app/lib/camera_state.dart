import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:io';

class CameraState with ChangeNotifier {
  XFile? _lastPictureTaken;
  bool _firstPictureTaken = false;

  /* getters */
  XFile? get lastPictureTaken => _lastPictureTaken;
  bool get firstPictureTaken => _firstPictureTaken;

  /* setters */
  set lastPicutureTaken(XFile? newValue) {
    _lastPictureTaken = newValue;
    _firstPictureTaken = true;
    notifyListeners();
  }
}
