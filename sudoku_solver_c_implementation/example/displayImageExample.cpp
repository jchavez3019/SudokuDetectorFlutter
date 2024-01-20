#include <iostream>
#include "SudokuDetector.h"

/*
This simple program takes a relative file path and displays the image. For example, you may enter a sudoku puzzle image from this repo
by running './displayImageExample  ../../../../sudoku.png'
*/

int main (int argc, char **argv) {
    if (argc < 2) {
        printf("Please enter a file path to an image\n");
        return -1;
    }
    printf("Using file path: %s\n", argv[1]);
    sudoku::displayImage(argv[1]);
    return 0;
}