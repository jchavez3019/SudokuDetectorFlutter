#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "helperFunctions.h"
#include <vector>
#include <cmath>
#include <tuple>

namespace sudoku {
    void displayImage(const char* image_path);

    int detectSudokuPuzzle(const char* image_path, cv::Mat &final_img, std::tuple<int, int> new_size, bool use_new_size = false, bool display = false);

    int detectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, std::tuple<int, int> new_size, bool use_new_size = false, bool display = false);
}