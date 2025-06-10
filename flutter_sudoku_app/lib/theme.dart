import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

var appTheme = ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: Colors.transparent, // Allows our animated background to show
    textTheme: TextTheme(
      bodyLarge: const TextStyle(
        fontSize: 18,
      ),
      bodyMedium: GoogleFonts.notoSans().copyWith(
        fontSize: 16,
        color: Colors.grey[300],
        backgroundColor: Colors.black.withValues(alpha: 0.3),
        fontWeight: FontWeight.bold,
      ),
      labelMedium: GoogleFonts.openSans().copyWith(
        fontSize: 16,
      ),
      labelLarge: GoogleFonts.openSans().copyWith(
        fontSize: 18,
        letterSpacing: 1.5,
        fontWeight: FontWeight.bold,
      ),
      displayLarge: const TextStyle(fontWeight: FontWeight.bold),
      titleMedium: const TextStyle(
        color: Colors.grey,
      ),
    ));
