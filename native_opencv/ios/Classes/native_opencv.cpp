#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ArucoDetector.h"
#include "SudokuDetector.h"

using namespace std;
using namespace cv;

static ArucoDetector* detector = nullptr;

void rotateMat(Mat &matImage, int rotation) {
    if (rotation == 90) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1);
    }
    else if (rotation == 270) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0);
    }
    else if (rotation == 180) {
        flip(matImage, matImage, -1);
    }
}

extern "C" {
// Attributes to prevent 'unused' function from being removed and to make it visible
__attribute__((visibility("default"))) __attribute__((used))
const char* version() {
    return CV_VERSION;
}

__attribute__((visibility("default"))) __attribute__((used))
void initDetector(uint8_t* markerPngBytes, int inBytesCount, int bits) {
    if (detector != nullptr) {
        delete detector;
        detector = nullptr;
    }

    vector<uint8_t> buffer(markerPngBytes, markerPngBytes + inBytesCount);
    Mat marker = imdecode(buffer, IMREAD_COLOR);

//    detector = new ArucoDetector(marker, bits);
}

__attribute((visibility("default"))) __attribute__((used))
void destroyDetector() {
    if (detector != nullptr) {
        delete detector;
        detector = nullptr;
    }
}

__attribute__((visibility("default"))) __attribute__((used))
uint8_t* rotateImage(uint8_t* originalImage, int inBytesCount, int* finalSize) {

    vector<uint8_t> buffer(originalImage, originalImage + inBytesCount);
    Mat cv_image = imdecode(buffer, IMREAD_COLOR);
    rotateMat(cv_image, 90);

    vector<uchar> buf;
    imencode(".jpg", cv_image, buf);
    *finalSize = buf.size() * sizeof(uchar);
    uint8_t* final_img = (uint8_t*)malloc(*finalSize);
    memcpy(final_img, (uint8_t*)buf.data(), *finalSize);
    return final_img;
}

/* Jorge Chavez
 * Description:
 * Inputs: 
 *      uint8_t* originalImage -- pointer to the original image
 *      int inBytesCount -- number of bytes the origina image takes
 *      int target_width -- target width of the final image 
 *      int target_height -- target height of the final image
 * Outputs:
 *      int* finalSize -- number of bytes the final image takes
*/
__attribute__((visibility("default"))) __attribute__((used))
uint8_t* detectSudokuPuzzle(uint8_t* originalImage, int inBytesCount, int target_width, int target_height, int* finalSize) {
    /* make a copy of the original image */
    vector<uint8_t> buffer(originalImage, originalImage + inBytesCount);
    /* load in the buffer as an opencv image */
    Mat cv_image = imdecode(buffer, IMREAD_COLOR);

    /* detect the sudoku puzzle */
    Mat sudoku_image;
    sudoku::detectSudokuPuzzle(cv_image, sudoku_image, make_tuple(target_width, target_height), true, false);

    /* return the final result */
    vector<uchar> buf;
    imencode(".jpg", sudoku_image, buf);
    *finalSize = buf.size() * sizeof(uchar);
    uint8_t* final_img = (uint8_t*)malloc(*finalSize);
    memcpy(final_img, (uint8_t*)buf.data(), *finalSize);
    return final_img;
}

/* Jorge Chavez
 * Description: Takes an image and applies a rotation
 * Inputs:
 *      int width -- width of the image
 *      int height -- height of the image
 *      int rotation -- rotation option for the image
 *      uint8_t* bytes -- pointer to the image data
 *      bool isYUV -- if the image is in YUV format, else is BGRA format
 * Outputs:
 *      int32_t* outCount -- size of the returning output image
 * Returns:
 *      float* jres -- pointer to the final transformed image
 * Effects:
 */
__attribute__ ((visibility("default"))) __attribute__((used))
const float* detect (int width, int height, int rotation, uint8_t* bytes, bool isYUV, int32_t* outCount) {
    if (detector == nullptr) {
        float* jres = new float[1];
        jres[0] = 0;
    }

    Mat frame;
    if (isYUV) {
        Mat myyuv(height + height / 2, width, CV_8UC1, bytes);
        cvtColor(myyuv, frame, COLOR_YUV2BGRA_NV21);
    }
    else {
        frame = Mat(height, width, CV_8UC4, bytes);
    }

    rotateMat(frame, rotation);
    cvtColor(frame, frame, COLOR_BGRA2GRAY);
    vector<ArucoResult> res = detector->detectArucos(frame, 0);

    vector<float> output;

    /* this returns a struct of 4 Point2f values */
    for (int i = 0; i < res.size(); i++) {
        ArucoResult ar = res[i];
        // each aruco has 4 corners
        output.push_back(ar.corners[0].x);
        output.push_back(ar.corners[0].y);
        output.push_back(ar.corners[1].x);
        output.push_back(ar.corners[1].y);
        output.push_back(ar.corners[2].x);
        output.push_back(ar.corners[2].y);
        output.push_back(ar.corners[3].x);
        output.push_back(ar.corners[3].y);
    }

    // Copy result bytes as output vec will get freed
    unsigned int total = sizeof(float) * output.size();
    float* jres = (float*)malloc(total);
    memcpy(jres, output.data(), total);

    *outCount = output.size();
    return jres;

}
}