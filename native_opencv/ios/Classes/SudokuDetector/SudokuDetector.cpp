#include "SudokuDetector.h"

using namespace std;
using namespace cv;

int _generalDetectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, std::tuple<int, int> new_size, bool use_new_size /* false */, bool display /* false */);

/**
 * @author Jorge Chavez
 * @brief reads in an image in color, but this function really doesn't do anything.
 * @param image_path: Path to the image to open.
 */
void sudoku::displayImage(const char* image_path) {
    Mat image;
    image = imread(image_path, IMREAD_COLOR);

    if (!image.data) {
        printf("No image data\n");
        return;
    }

    return;
}

/**
 * @author  Jorge Chavez
 * @brief   Given a path to an image, runs our vision processing pipeline in order to get a bird's
 *          eye view of the sudoku board.
 * @param image_path:   Path to the image to open.
 * @param final_img:    Holds the final processed image.
 * @param new_size:     The size the final image should take if `use_new_size` is true.
 * @param use_new_size: If True, the final image is resized to the target size, `new_size`.
 * @param display:      If True, displays the transformed images. Supported on desktop, not Android.
 * @return
 */
int sudoku::detectSudokuPuzzle(const char *image_path, Mat &final_img, std::tuple<int, int> new_size, bool use_new_size /* false */, bool display /* false */) {
    Mat original_image;
    
    /* original image to never be modified */
    original_image = imread(image_path, IMREAD_COLOR);

    return _generalDetectSudokuPuzzle(original_image, final_img, new_size, use_new_size, display);
}

/**
 * @author  Jorge Chavez
 * @brief   Given a CV image, runs our vision processing pipeline in order to get a bird's
 *          eye view of the sudoku board.
 * @param original_image:   CV image to process.
 * @param final_img:        Holds the final processed image.
 * @param new_size:         The size the final image should take if `use_new_size` is true.
 * @param use_new_size:     If True, the final image is resized to the target size, `new_size`.
 * @param display:          If True, displays the transformed images. Supported on desktop, not Android.
 * @return
 */
int sudoku::detectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, std::tuple<int, int> new_size, bool use_new_size /* false */, bool display /* false */) {
    return _generalDetectSudokuPuzzle(original_image, final_img, new_size, use_new_size, display);
}

/**
 * @author  Jorge Chavez
 * @brief   Resizes an image to the target size with cubic interpolation for better quality.
 * @param original_image:   Original image to resize
 * @param final_img:        Holds the final resized image.
 * @param target_width:     The target width to resize to.
 * @param target_height:    The target height to resize to.
 * @return
 */
int _resize_image_high_quality(const cv::Mat &original_image, cv::Mat &final_img, int target_width=3400, int target_height=2500) {
    // get the dimensions of the original image
    int img_width = original_image.cols;
    int img_height = original_image.rows;

    if (img_width >= target_width && img_height >= target_height) {
        final_img = original_image.clone();
        return 0;
    }

    cv::resize(original_image, final_img, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    return 0;
}

/**
 * @author  Jorge Chavez
 * @brief   Given a vector of 4 points, reorder them to be in the order: top left, top right,
 *          bottom right, bottom left.
 * @param original_pts:     A vector of 4 pixel coordinates.
 * @param new_order_pts:    Will holds the final pixel coordinates in the desired order.
 * @return
 */
int _reorder_points(const vector<Point> &original_pts, vector<Point> &new_order_pts) {

    // calculate the magnitude of all the points
    vector<int> abs_vals(4, 0);
    int i, abs_val;
    for (i = 0; i < 4; i++) {
        abs_vals[i] = original_pts[i].x * original_pts[i].y;
    }
    // initialize our four points
    Point top_left(-1, -1);
    Point bottom_right(-1, -1);
    Point top_right(-1, -1);
    Point bottom_left(-1, -1);

    // get the indices and coordinates of the top left and bottom right pixels
    int tl_idx, br_idx, o_c;
    for (i = 0; i < 4; i++) {
        abs_val = abs_vals[i];
        if (top_left.x == -1 || abs_val < top_left.x * top_left.y) {
            top_left = original_pts[i];
            tl_idx = i;
        }
        if (bottom_right.x == -1 || abs_val > bottom_right.x * bottom_right.y) {
            bottom_right = original_pts[i];
            br_idx = i;
        }
    }

    // get the indices of the remaining points
    vector<int> other_idx(2, 0);
    o_c = 0;
    for (i = 0; i < 4; i++) {
        if (i != tl_idx && i != br_idx) {
            other_idx[o_c] = i;
            o_c++;
        }
    }

    // extract the remaing top right and bottom left points
    if (original_pts[other_idx[0]].y > original_pts[other_idx[1]].y) {
        top_right = original_pts[other_idx[0]];
        bottom_left = original_pts[other_idx[1]];
    }
    else {
        top_right = original_pts[other_idx[1]];
        bottom_left = original_pts[other_idx[0]];
    }

    // return the points in the desired order
    new_order_pts[0] = top_left;
    new_order_pts[1] = top_right;
    new_order_pts[2] = bottom_right;
    new_order_pts[3] = bottom_left;

    return 0;
}

/**
 * @author  Jorge Chavez
 * @brief   The main sudoku image processing pipeline.
 * @param original_image:   CV image to process.
 * @param final_img:        Holds the final processed image.
 * @param new_size:         The size the final image should take if `use_new_size` is true.
 * @param use_new_size:     If True, the final image is resized to the target size, `new_size`.
 * @param display:          If True, displays the transformed images. Supported on desktop, not Android.
 * @return
 */
int _generalDetectSudokuPuzzle(cv::Mat &original_image, cv::Mat &final_img, std::tuple<int, int> new_size, bool use_new_size /* false */, bool display /* false */) {
    Mat colored_image, o_gray_image, gray_image; // original image, original gray image, resized gray image

    /* gray scale copy of the original image */
    cvtColor(original_image, o_gray_image, COLOR_BGR2GRAY);

    /* resize the image */
    _resize_image_high_quality(o_gray_image, gray_image);
    _resize_image_high_quality(original_image, colored_image);

    /* colored copy of the original image */
    // colored_image = original_image.clone(); // redundant copy
    int img_width = gray_image.cols; int img_height = gray_image.rows;
    int img_area = img_width * img_height;
    const float area_threshold = 0.98;

    if (!gray_image.data) {
        printf("No image data\n");
        return -1;
    }

    Mat modified_image; // this will be getting modified throughout the program

    /* apply a median blur to the image */
    medianBlur(gray_image, modified_image, 25);

    /* apply adaptive thresholding */
    adaptiveThreshold(modified_image, modified_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);

    /* apply the Canny edge detector */
    Canny(modified_image, modified_image, 250, 200);

    /* apply dilation */
    Mat kernel;
    kernel = Mat::ones(11, 11, CV_8U);
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
    printf("Largest areas index is %d with area %f\nNumber of Contours: %zu\n", area_indices[0], contours_areas[area_indices[0]], contours.size());

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
    vector<Point> o_approx; // original coordinates of the convex hull approximation
    vector<Point> approx(4, Point(-1, -1));
    // double perimeter = arcLength(chosen_contour, true) * 0.02;
    // approxPolyDP(chosen_contour, approx, perimeter, true);
    approxPolyN(chosen_contour, o_approx, 4, -1.0, true);

    // reorder the approximate points to be in the order: top left, top right, bottom right,
    // bottom left
    _reorder_points(o_approx, approx);

    /* check that the modified contour has valid properties */
    // double approx_area = contourArea(approx);
    // if (!(approx.size() == 4 && fabs(approx_area) > 2000.0 && isContourConvex(approx))) {
    //     printf("Approximate contour did not satisfy requirements\n");
    //     return -1;
    // }

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
    Size final_size(square_size, square_size);
    warpPerspective(colored_image, final_img, transformation_matrix, final_size);

    // resize the final image
    if (use_new_size) {
        cv::resize(final_img, final_img, cv::Size(std::get<0>(new_size), std::get<1>(new_size)), 0, 0, cv::INTER_CUBIC);
    }

    return 0;
}