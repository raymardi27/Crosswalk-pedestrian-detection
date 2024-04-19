#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat convertToLBP(const Mat& src_gray) {
    Mat lbpImg = src_gray.clone();
    for (int i = 1; i < src_gray.rows - 1; i++) {
        for (int j = 1; j < src_gray.cols - 1; j++) {
            uchar center = src_gray.at<uchar>(i, j);
            unsigned char lbp = 0;
            lbp |= (src_gray.at<uchar>(i - 1, j - 1) > center) << 7;
            lbp |= (src_gray.at<uchar>(i - 1, j) > center) << 6;
            lbp |= (src_gray.at<uchar>(i - 1, j + 1) > center) << 5;
            lbp |= (src_gray.at<uchar>(i, j + 1) > center) << 4;
            lbp |= (src_gray.at<uchar>(i + 1, j + 1) > center) << 3;
            lbp |= (src_gray.at<uchar>(i + 1, j) > center) << 2;
            lbp |= (src_gray.at<uchar>(i + 1, j - 1) > center) << 1;
            lbp |= (src_gray.at<uchar>(i, j - 1) > center) << 0;
            lbpImg.at<uchar>(i, j) = lbp;
        }
    }
    return lbpImg;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <VideoPath>" << endl;
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame, hsv, skinMask, gray, lbp;
    while (cap.read(frame)) {
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(0, 48, 80), Scalar(20, 255, 255), skinMask);

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        lbp = convertToLBP(gray);

        // Apply the skin mask to the LBP image
        Mat maskedLBP;
        lbp.copyTo(maskedLBP, skinMask);

        // Find contours in the skin mask
        vector<vector<Point>> contours;
        findContours(skinMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Filter contours and draw rectangles around potential faces
        for (const auto& contour : contours) {
            Rect rect = boundingRect(contour);
            float faceLikeRatio = static_cast<float>(rect.width) / rect.height;

            // Check for reasonable face-like aspect ratios and size
            if (faceLikeRatio > 0.5 && faceLikeRatio < 1.5 && rect.area() > 1000) {
                rectangle(frame, rect, Scalar(0, 255, 0), 2);
            }
        }

        imshow("Frame", frame);
        imshow("Skin Mask", skinMask);
        imshow("Masked LBP", maskedLBP);

        if (waitKey(30) == 27) break; // Exit on ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

