import 'package:flutter/material.dart';

class Loader extends StatelessWidget {
  const Loader({super.key});

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final loaderSize = size.width * 0.25;

    return SizedBox(
      width: loaderSize,
      height: loaderSize,
      child: const CircularProgressIndicator(
        strokeWidth: 6.0, // optional: make it more visible
      ),
    );
  }
}


class LoadingScreen extends StatelessWidget {
  const LoadingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
        title: const Text('Loading...'),
      backgroundColor: Colors.deepPurple,
        ),
      body: const Center(
        child: Loader(),
      ),
    );
  }
}
