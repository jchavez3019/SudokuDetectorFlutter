#pragma once
#include <opencv4/opencv2/opencv.hpp>
#include "helperFunctions.h"
#include <vector>
#include <cmath>

namespace sudoku {
    void displayImage(const char* image_path);

    int detectSudokuPuzzle(const char* image_path, cv::Mat &final_img, bool display = false);
}