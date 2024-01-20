#include <iostream>
#include "SudokuDetector.h"

using namespace cv;

/*
This simple program takes a relative file path and displays the image. For example, you may enter a sudoku puzzle image from this repo
by running 
'./detectSudokuExample  ../../../../sudoku.png'
'./detectSudokuExample ../../../../sudoku_newspaper.jpeg'
'./detectSudokuExample ../../../../sudoku_newspaper2.jpg'
*/

int main (int argc, char **argv) {
    if (argc < 2) {
        printf("Please enter a file path to an image\n");
        return -1;
    }

    /* first display the image so that the user can take a look at the image before it is processed */
    printf("Using file path: %s\n", argv[1]);
    Mat final_img;
    int result = sudoku::detectSudokuPuzzle(argv[1], final_img, false);
    printf("Returned from sudoku::detectSudokuPuzzle with result %d\n", result);
    namedWindow("Final Image", WINDOW_AUTOSIZE);
    imshow("Final Image", final_img);
    waitKey(0);

    return 0;
}