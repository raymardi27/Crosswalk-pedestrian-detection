#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Function to convert a color image into an LBP image
Mat convertToLBP(const Mat& src_gray) {
    Mat lbpImg = src_gray.clone();
    for(int i = 1; i < src_gray.rows - 1; i++) {
        for(int j = 1; j < src_gray.cols - 1; j++) {
            uchar center = src_gray.at<uchar>(i,j);
            unsigned char lbp = 0;
            lbp |= (src_gray.at<uchar>(i-1, j-1) > center) << 7;
            lbp |= (src_gray.at<uchar>(i-1, j  ) > center) << 6;
            lbp |= (src_gray.at<uchar>(i-1, j+1) > center) << 5;
            lbp |= (src_gray.at<uchar>(i,   j+1) > center) << 4;
            lbp |= (src_gray.at<uchar>(i+1, j+1) > center) << 3;
            lbp |= (src_gray.at<uchar>(i+1, j  ) > center) << 2;
            lbp |= (src_gray.at<uchar>(i+1, j-1) > center) << 1;
            lbp |= (src_gray.at<uchar>(i,   j-1) > center) << 0;
            lbpImg.at<uchar>(i,j) = lbp;
        }
    }
    return lbpImg;
}

// Main function
int main() {
    VideoCapture cap(); // Open the default camera
    if (!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame, hsv, skinMask, gray, lbp;
    while (cap.read(frame)) {
        // Convert frame to HSV color space
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Skin color range for detection in HSV
        inRange(hsv, Scalar(0, 48, 80), Scalar(20, 255, 255), skinMask);

        // Convert frame to grayscale for LBP
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Get LBP image
        lbp = convertToLBP(gray);

        // Apply the skin mask to the LBP image
        lbp &= skinMask;

        // Simple threshold to identify regions, this is a crude approximation
        // and would normally require more sophisticated processing.
        threshold(lbp, lbp, 50, 255, THRESH_BINARY);

        // Display results
        imshow("Frame", frame);
        imshow("Skin Mask", skinMask);
        imshow("LBP Image", lbp);

        if (waitKey(30) >= 0) break; // Wait for any key press
    }

    // When everything done, release the video capture object
    cap.release();
    destroyAllWindows();

    return 0;
}
