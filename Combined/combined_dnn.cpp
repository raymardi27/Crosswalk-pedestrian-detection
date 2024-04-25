#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#define Combined_Condition classIdPoint.x == 1 || classIdPoint.x == 0

using namespace cv;
using namespace cv::dnn;
using namespace std;

const float FACE_CONFIDENCE_THRESHOLD = 0.3;
const float PERSON_CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int NETWORK_WIDTH = 416;
const int NETWORK_HEIGHT = 416;

// Blur the face
// void Blur(int toBlur, float conf, int left, int top, int right, int bottom, Mat& frame) {
void Blur(int toBlur, int left, int top, int right, int bottom, Mat& frame) {

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);

    if(toBlur){
        // Define the region of interest
        cv::Rect roi(left, top, right - left + 1, bottom - top + 1);

        // Check if the ROI is within the frame bounds
        if (roi.x >= 0 && roi.y >= 0 && roi.width + roi.x <= frame.cols && roi.height + roi.y <= frame.rows) {
            // Extract the ROI from the frame
            cv::Mat roiImg = frame(roi);

            // Apply Gaussian blur to the extracted ROI
            cv::GaussianBlur(roiImg, roiImg, cv::Size(31, 31), 10.0, 10.0);
        
            // Copy the blurred ROI back to the original image
            roiImg.copyTo(frame(roi));
        } else {
            std::cerr << "Error: ROI is out of bounds." << std::endl;
        }
    }
    
    // string label = format("%.2f", conf);
    // label = "Face: " + label;
    // putText(frame, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}

void postprocess(Mat& frame, const vector<Mat>& outs,bool toBlur = false) {
    // vector<int> classIds;
    float CONFIDENCE_THRESHOLD = PERSON_CONFIDENCE_THRESHOLD;
    if(toBlur){
        CONFIDENCE_THRESHOLD = FACE_CONFIDENCE_THRESHOLD;
    }
    
    vector<float> confidences;
    double confidence;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence > CONFIDENCE_THRESHOLD && (Combined_Condition)){
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                // classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        // Blur(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
        Blur(toBlur,  box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <faces_cfg_file_path> <person_cfg_file_path> <face_weights_path> <person_weights_path> <video_path>" << endl;
        return 1;
    }

    String face_cfg_file = argv[1], 
            person_cfg_file = argv[2],
            faces_weights_file = argv[3],
            person_weights_file = argv[4];

    // Load network
    Net Facenet = readNet(face_cfg_file, faces_weights_file);
    if (Facenet.empty()) {
        cerr << "Could not load the face neural network. Check the weights and cfg file paths\n";
        return 1;
    }

    Net Personnet = readNet(person_cfg_file, person_weights_file);
    if (Personnet.empty()) {
        cerr << "Could not load the person neural network. Check the weights and cfg file paths\n";
        return 1;
    }
    
    // Check if OpenCV is built with CUDA support and set CUDA as preferable backend and target
    if (cuda::getCudaEnabledDeviceCount() > 0) {
        Facenet.setPreferableBackend(DNN_BACKEND_CUDA);
        Facenet.setPreferableTarget(DNN_TARGET_CUDA);
        Personnet.setPreferableBackend(DNN_BACKEND_CUDA);
        Personnet.setPreferableTarget(DNN_TARGET_CUDA);
        cout << "Using CUDA for processing\n";
    } else {
        cerr << "CUDA not available on this device; using CPU.\n";
        Facenet.setPreferableBackend(DNN_BACKEND_OPENCV);
        Facenet.setPreferableTarget(DNN_TARGET_CPU);
        Personnet.setPreferableBackend(DNN_BACKEND_OPENCV);
        Personnet.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Open a video file or a camera stream.
    VideoCapture cap(argv[5]);
    if (!cap.isOpened()) {
        cerr << "Could not open video " << argv[1] << endl;
        return 1;
    }

    cv::namedWindow("Detect", cv::WINDOW_NORMAL);  // Create a window that can be resized
    cv::resizeWindow("Detect", 1280, 720);          // Set the dimensions of the window

    // Define the classes we aim to detect
    vector<string> classes = {"person","face"};


    Mat frame, blob;
    double fps = 0.0;
    int frameCnt = 0;
    // double startTime = (double)getTickCount(); // start fps
    double t,duration,seconds;
    string label;
    while (cap.read(frame)) {
        t = (double)getTickCount();
        blobFromImage(frame, blob, 1/255.0, Size(NETWORK_WIDTH, NETWORK_HEIGHT), Scalar(0, 0, 0), true, false);
        Facenet.setInput(blob);
        Personnet.setInput(blob);
        vector<Mat> outs;
        Personnet.forward(outs, Personnet.getUnconnectedOutLayersNames());
        postprocess(frame, outs);

        Facenet.forward(outs, Facenet.getUnconnectedOutLayersNames());
        postprocess(frame, outs,true);

        ++frameCnt;
        duration = (double)getTickCount() - t;
        seconds = duration/getTickFrequency();
        fps = 1.0/seconds;

        // Display FPS on frame
        string label = format("FPS: %.2f", fps);
        putText(frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

        imshow("Detect", frame);
        if (waitKey(1) == 27) break; // stop if escape key is pressed
    }
    cap.release();
    destroyAllWindows();
    return 0;
}