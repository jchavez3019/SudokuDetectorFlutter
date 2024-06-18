#include <iostream>
#include "SudokuDetector.h"

/*
This simple program takes a relative file path and displays the image. For example, you may enter a sudoku puzzle image from this repo
by running './displayImageExample  ../../../sudoku.png'
*/

char default_image_path [] = "../../../sudoku.png";

int main (int argc, char **argv) {
    char* img_path;
    if (argc < 2) {
        printf("Using default image path %s\n", default_image_path);
        img_path = default_image_path;
    }
    else {
        img_path = argv[1];
    }
    printf("Using file path: %s\n", img_path);
    sudoku::displayImage(img_path);
    return 0;
}