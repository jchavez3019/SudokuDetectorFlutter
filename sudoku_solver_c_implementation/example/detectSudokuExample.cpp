#include <iostream>
#include "SudokuDetector.h"

using namespace cv;

/*
This simple program takes a relative file path and displays the image. For example, you may enter a sudoku puzzle image from this repo
by running 
'./detectSudokuExample  ../../../sudoku.png'
'./detectSudokuExample ../../../sudoku_newspaper.jpeg'
'./detectSudokuExample ../../../sudoku_newspaper2.jpg'
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

    /* first display the image so that the user can take a look at the image before it is processed */
    printf("Using file path: %s\n", img_path);
    Mat final_img;
    int result = sudoku::detectSudokuPuzzle(img_path, final_img, false);
    printf("Returned from sudoku::detectSudokuPuzzle with result %d\n", result);
    namedWindow("Final Image", WINDOW_AUTOSIZE);
    imshow("Final Image", final_img);
    waitKey(0);

    /* now check the function overload */
    Mat original_image = imread(img_path, IMREAD_COLOR);
    result = sudoku::detectSudokuPuzzle(original_image, final_img, false);
    printf("Returned from sudoku::detectSudokuPuzzle with result %d\n", result);
    namedWindow("Final Image", WINDOW_AUTOSIZE);
    imshow("Final Image", final_img);
    waitKey(0);

    return 0;
}