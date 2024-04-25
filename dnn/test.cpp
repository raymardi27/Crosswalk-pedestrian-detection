#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	cout<< "Cuda Device Count: "<< cuda::getCudaEnabledDeviceCount()<<endl;
	return 0;
}

