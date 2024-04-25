#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main(int argc, char** argv) {
    
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <cfg_file_path> <weights_path> <video_path>" << endl;
        return 1;
    }

    String cfg_file = argv[1], weights_file = argv[2];

    // Load network
    Net net = readNet(cfg_file, weights_file);
    if (net.empty()) {
        cerr << "Could not load the neural network. Check the weights and cfg file paths\n";
        return 1;
    }
    
    // Set backend and target to CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // Open the video file
    cv::VideoCapture cap(argv[3]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file " << argv[1] << std::endl;
        return 1;
    }

    // Load class names from file
    std::vector<std::string> classes;
    std::ifstream ifs("coco.names");
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Create a blob from the image and use it as input to the network
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(), true, false);
        net.setInput(blob);

        // Forward pass through the network
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Process outputs
        for (auto& output : outs) {
            float* data = (float*)output.data;
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
                if (confidence > 0.4 && classIdPoint.x == 0) {  // Check if detected class is "person"
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    cv::rectangle(frame, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 3);
                    std::cout << "Detected person with confidence: " << confidence * 100 << "%" << std::endl;
                }
            }
        }

        // Display the frame
        cv::imshow("Person Detection", frame);
        if (cv::waitKey(1) == 27) break;  // Stop if ESC key is pressed
    }

    return 0;
}
