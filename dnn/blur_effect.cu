#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void blurKernel(int left, int top, int right, int bottom, uchar3* input, uchar3* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + left;
    int row = blockIdx.y * blockDim.y + threadIdx.y + top;

    if (col < left || col >= right || row < top || row >= bottom) return;

    int count = 0;
    float sumB = 0, sumG = 0, sumR = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int currentX = col + j;
            int currentY = row + i;
            if (currentX >= left && currentX < right && currentY >= top && currentY < bottom) {
                uchar3 pixel = input[currentY * width + currentX];
                sumB += pixel.x;
                sumG += pixel.y;
                sumR += pixel.z;
                count++;
            }
        }
    }

    uchar3& outPixel = output[row * width + col];
    outPixel.x = static_cast<unsigned char>(sumB / count);
    outPixel.y = static_cast<unsigned char>(sumG / count);
    outPixel.z = static_cast<unsigned char>(sumR / count);
}

void cudaBlur(cv::Mat& frame, int left, int top, int right, int bottom) {
    // Ensure correct dimensions
    left = std::max(0, left);
    right = std::min(frame.cols, right);
    top = std::max(0, top);
    bottom = std::min(frame.rows, bottom);

    // Image dimensions
    const int width = frame.cols;
    const int height = frame.rows;

    size_t bytes = width * height * sizeof(uchar3);
    uchar3 *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy data to device
    cudaMemcpy(d_input, frame.data, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 numBlocks((right - left + blockSize.x - 1) / blockSize.x, (bottom - top + blockSize.y - 1) / blockSize.y);

    blurKernel<<<numBlocks, blockSize>>>(left, top, right, bottom, d_input, d_output, width, height);

    // Copy result back to host
    cudaMemcpy(frame.data, d_output, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
