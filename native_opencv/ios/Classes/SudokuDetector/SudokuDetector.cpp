#include "SudokuDetector.h"

using namespace std;
using namespace cv;

int _generalDetectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, bool display = false);

void sudoku::displayImage(const char* image_path) {
    Mat image;
    image = imread(image_path, IMREAD_COLOR);

    if (!image.data) {
        printf("No image data\n");
        return;
    }

    return;
}

int sudoku::detectSudokuPuzzle(const char *image_path, Mat &final_img, bool display /* false */) {
    Mat original_image;
    
    /* original image to never be modified */
    original_image = imread(image_path, IMREAD_COLOR);

    return _generalDetectSudokuPuzzle(original_image, final_img, display);
}

int sudoku::detectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, bool display /* false */) {
    return _generalDetectSudokuPuzzle(original_image, final_img, display);
}

int _generalDetectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, bool display /* false */) {
        Mat colored_image, gray_image;

    /* gray scale copy of the original image */
    cvtColor(original_image, gray_image, COLOR_BGRA2GRAY);

    /* colored copy of the original image */
    colored_image = original_image.clone();
    int img_width = gray_image.cols; int img_height = gray_image.rows;
    int img_area = img_width * img_height;
    const float area_threshold = 0.98;

    if (!gray_image.data) {
        printf("No image data\n");
        return -1;
    }

    Mat modified_image; // this will be getting modified throughout the program

    /* apply a median blur to the image */
    medianBlur(gray_image, modified_image, 3);

    /* apply adaptive thresholding */
    adaptiveThreshold(modified_image, modified_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);

    /* apply the Canny edge detector */
    Canny(modified_image, modified_image, 250, 200);

    /* apply dilation */
    Mat kernel;
    kernel = Mat::ones(3, 3, CV_8U);
    dilate(modified_image, modified_image, kernel);

    /* extract the contours */
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(modified_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    /* order the contours from largest area to smallest */
    vector<double> contours_areas(contours.size());
    vector<int> area_indices(contours.size());
    for (int i = 0; i < contours.size(); i++)
        contours_areas[i] = contourArea(contours[i]);

    helper::argMergeSort<double>(&contours_areas[0], &area_indices[0], 0, contours.size() - 1);
    helper::flipArray(&area_indices[0], contours.size());
    printf("Largest areas index is %d with area %f\nNumber of Contours: %ld\n", area_indices[0], contours_areas[area_indices[0]], contours.size());

    /* filter out contours whose area is nearly the entire image as */
    /* this would indicate that there is a border around the image */
    int c_start = 0; double curr_area;
    for (int i = 0; i < contours.size(); i++) {
        curr_area = contours_areas[area_indices[i]];
        if (curr_area / img_area < area_threshold) {
            c_start = i;
            break;
        }
    }

    /* approximate the biggest contour into a nicer polygon shape */
    vector<Point> chosen_contour(contours[area_indices[c_start]]);
    vector<Point> approx;
    double perimeter = arcLength(chosen_contour, true) * 0.02;
    approxPolyDP(chosen_contour, approx, perimeter, true);

    /* check that the modified contour has valid properties */
    double approx_area = contourArea(approx);
    if (!(approx.size() == 4 && fabs(approx_area) > 2000.0 && isContourConvex(approx))) {
        printf("Approximate contour did not satisfy requirements\n");
        return -1;
    }
    /* apply a homography transformation so that we have a centered square that takes */
    /* most of the image of the Sudoku Puzzle */
    double square_size;
    if (img_height < img_width)
        square_size = floor(img_height * 0.90);
    else
        square_size = floor(img_width * 0.90);

    vector<Point2f> target_coordinates = {
        Point2f(0.0, 0.0),
        Point2f(0.0, square_size),
        Point2f(square_size, square_size),
        Point2f(square_size, 0.0)
    };
    vector<Point2f> approx_2f;
    Mat(approx).copyTo(approx_2f);
    Mat transformation_matrix = getPerspectiveTransform(approx_2f, target_coordinates);
    // Mat final_img;
    Size final_size(square_size, square_size);
    colored_image = original_image.clone();
    warpPerspective(colored_image, final_img, transformation_matrix, final_size);

    return 0;
}