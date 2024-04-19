#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int NETWORK_WIDTH = 416;
const int NETWORK_HEIGHT = 416;

vector<string> getClasses(const string& classesFile) {
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    return classes;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
    string label = format("%.2f", conf);
    label = "Face: " + label;
    putText(frame, label, Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}

void postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            float confidence = data[4];
            if (confidence > CONFIDENCE_THRESHOLD) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(0);
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
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;
        return 1;
    }

    // Load network
    Net net = readNet("tiny-yolo-widerface.cfg", "tiny-yolo-widerface_final.weights");
    if (net.empty()) {
        cerr << "Could not load the neural network.\n";
        return 1;
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open a video file or a camera stream.
    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Could not open video " << argv[1] << endl;
        return 1;
    }

    Mat frame, blob;
    while (cap.read(frame)) {
        blobFromImage(frame, blob, 1/255.0, Size(NETWORK_WIDTH, NETWORK_HEIGHT), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        postprocess(frame, outs);

        imshow("Face Detection", frame);
        if (waitKey(1) == 27) break; // stop if escape key is pressed
    }
    cap.release();
    destroyAllWindows();
    return 0;
}

