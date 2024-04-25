#ifndef BLUR_EFFECT_H
#define BLUR_EFFECT_H

#include <opencv2/core/core.hpp>

// Function declaration
void cudaBlur(cv::Mat& frame, int left, int top, int right, int bottom);

#endif // BLUR_EFFECT_H
