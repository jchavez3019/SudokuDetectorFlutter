import 'dart:math';
import 'package:flutter/material.dart';

class BackgroundAnimation extends StatefulWidget {
  const BackgroundAnimation({super.key});

  @override
  State<BackgroundAnimation> createState() => _BackgroundAnimationState();
}

class _BackgroundAnimationState extends State<BackgroundAnimation>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  final Random _random = Random();

  static const animationDuration = Duration(seconds: 1200);

  @override
  void initState() {
    super.initState();
    _controller =
    AnimationController(vsync: this, duration: animationDuration)
      ..repeat();
  }

  Widget _movingShape(Widget child, double dx, double dy, Duration offset) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (_, __) {
        final t = (_controller.value + offset.inMilliseconds/animationDuration.inMilliseconds) % 1;
        final y = sin(t * 2 * pi)*dy;
        final x = cos(t * 2 * pi)*dx;

        return Transform.translate(offset: Offset(x, y), child: child);
      },
    );
  }

  Widget _digit() {
    final digit = (_random.nextInt(9) + 1).toString();
    final size = _random.nextDouble() * 40 + 20;
    final color = Colors.deepPurpleAccent.withValues(alpha: 0.1 + _random.nextDouble() * 0.1);

    return Positioned(
      top: _random.nextDouble() * MediaQuery.of(context).size.height,
      left: _random.nextDouble() * MediaQuery.of(context).size.width,
      child: _movingShape(
        DefaultTextStyle(
          style: TextStyle(
              fontSize: size, color: color, fontWeight: FontWeight.bold),
          child: Text(
            digit,
          ),
        ),
        10 + _random.nextDouble() * 20,
        10 + _random.nextDouble() * 20,
        Duration(milliseconds: _random.nextInt(20000)),
      ),
    );
  }

  Widget _square() {
    final size = 20.0 + _random.nextDouble() * 30;

    return Positioned(
      top: _random.nextDouble() * MediaQuery.of(context).size.height,
      left: _random.nextDouble() * MediaQuery.of(context).size.width,
      child: _movingShape(
        Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            color: Colors.transparent, // No fill
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: Colors.indigo.withOpacity(0.5), // Border color
              width: 4.0,
            ),
          ),
        ),
        20,
        20,
        Duration(milliseconds: _random.nextInt(20000)),
      ),
    );
  }


  Widget _cross() {
    const size = 40.0;
    final color = Colors.blue.withValues(alpha: 0.08);

    return Positioned(
      top: _random.nextDouble() * MediaQuery.of(context).size.height,
      left: _random.nextDouble() * MediaQuery.of(context).size.width,
      child: _movingShape(
        Transform.rotate(
          angle: _controller.value * 2 * pi,
          child: CustomPaint(size: const Size(size, size), painter: CrossPainter(color)),
        ),
        15,
        15,
        Duration(milliseconds: _random.nextInt(20000)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: SizedBox.expand(
        child: Stack(
          children: [
            ...List.generate(8, (_) => _digit()),
            ...List.generate(5, (_) => _square()),
            ...List.generate(5, (_) => _cross()),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class CrossPainter extends CustomPainter {
  final Color color;
  CrossPainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 6
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(const Offset(0, 0), Offset(size.width, size.height), paint);
    canvas.drawLine(Offset(size.width, 0), Offset(0, size.height), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
