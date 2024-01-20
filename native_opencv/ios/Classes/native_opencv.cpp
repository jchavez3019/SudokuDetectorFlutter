#include "ArucoDetector.h"

using namespace std;
using namespace cv;

static ArucoDetector* detector = nullptr;

const char hello_world[] = "Hello world from jorgejc2";

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
}

extern "C" {
// Attributes to prevent 'unused' function from being removed and to make it visible
__attribute__((visibility("default"))) __attribute__((used))
const char* hello() {
    return hello_world;
}
}