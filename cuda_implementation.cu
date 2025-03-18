#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;

// ---------------------------------------------------------------------------
// CUDA Error Check Helper
// ---------------------------------------------------------------------------
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "Error: " << msg << " " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
// Convolution Kernel: Applies a convolution filter to a 3-channel image.
// The image is stored in row-major order (H x W x C) with C=3.
// ---------------------------------------------------------------------------
__global__ void convolution_kernel(const float* input, const float* kernel, float* output,
                                   int height, int width, int channels, int kernel_size) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;
    if(pid >= total_pixels) return;
    
    int row_idx = pid / width;
    int col_idx = pid % width;
    int half_kernel = kernel_size / 2;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Loop over kernel rows and columns
        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                int in_row = row_idx + kr - half_kernel;
                int in_col = col_idx + kc - half_kernel;
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    int input_offset = (in_row * width + in_col) * channels + c;
                    int kernel_offset = kr * kernel_size + kc;
                    acc += input[input_offset] * kernel[kernel_offset];
                }
            }
        }
        int output_offset = (row_idx * width + col_idx) * channels + c;
        output[output_offset] = acc;
    }
}

// ---------------------------------------------------------------------------
// Resize Kernel: Resizes an image using bilinear interpolation.
// Assumes a single image with shape (in_height x in_width x channels).
// ---------------------------------------------------------------------------
__global__ void resize_kernel(const float* input, float* output,
                              int in_height, int in_width, int channels,
                              int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = out_height * out_width;
    if (idx >= total_pixels) return;
    
    int out_row = idx / out_width;
    int out_col = idx % out_width;
    
    float scale_y = float(in_height) / out_height;
    float scale_x = float(in_width) / out_width;
    
    float in_y = out_row * scale_y;
    float in_x = out_col * scale_x;
    
    int in_y0 = (int)in_y;
    int in_x0 = (int)in_x;
    int in_y1 = min(in_y0 + 1, in_height - 1);
    int in_x1 = min(in_x0 + 1, in_width - 1);
    
    float wy1 = in_y - in_y0;
    float wx1 = in_x - in_x0;
    float wy0 = 1.0f - wy1;
    float wx0 = 1.0f - wx1;
    
    for (int c = 0; c < channels; c++) {
        int top_left_offset = (in_y0 * in_width + in_x0) * channels + c;
        int top_right_offset = (in_y0 * in_width + in_x1) * channels + c;
        int bottom_left_offset = (in_y1 * in_width + in_x0) * channels + c;
        int bottom_right_offset = (in_y1 * in_width + in_x1) * channels + c;
        
        float top_left = input[top_left_offset];
        float top_right = input[top_right_offset];
        float bottom_left = input[bottom_left_offset];
        float bottom_right = input[bottom_right_offset];
        
        float value = top_left * (wx0 * wy0) + top_right * (wx1 * wy0) +
                      bottom_left * (wx0 * wy1) + bottom_right * (wx1 * wy1);
        
        int output_offset = (out_row * out_width + out_col) * channels + c;
        output[output_offset] = value;
    }
}

// ---------------------------------------------------------------------------
// CudaImageProcessor Class: Provides image processing functions using CUDA.
// Uses OpenCV for image I/O and performs operations such as convolution,
// Gaussian blur, Sobel edge detection, sharpening, and resizing.
// ---------------------------------------------------------------------------
class CudaImageProcessor {
public:
    CudaImageProcessor() {}
    
    // Load an image (BGR) and convert to float in range [0,1]
    cv::Mat loadImage(const std::string& path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Error: Unable to load image " << path << endl;
            exit(EXIT_FAILURE);
        }
        img.convertTo(img, CV_32FC3, 1.0/255.0);
        return img;
    }
    
    // Save an image (assumed to be CV_32FC3) after converting to 8-bit
    void saveImage(const cv::Mat& img, const std::string& path) {
        cv::Mat out;
        img.convertTo(out, CV_8UC3, 255.0);
        cv::imwrite(path, out);
    }
    
    // Apply a convolution kernel to an image using the CUDA convolution kernel.
    // The kernel is assumed to be a square matrix (CV_32F).
    cv::Mat applyKernel(const cv::Mat& image, const cv::Mat& kernel) {
        int height = image.rows;
        int width = image.cols;
        int channels = image.channels();
        int kernel_size = kernel.rows; // assume square kernel
        
        size_t imgSize = height * width * channels * sizeof(float);
        size_t kernelSize = kernel_size * kernel_size * sizeof(float);
        
        // Ensure image is continuous
        cv::Mat imageFloat;
        if (!image.isContinuous())
            image.copyTo(imageFloat);
        else
            imageFloat = image;
        
        // Output image
        cv::Mat output = cv::Mat::zeros(height, width, CV_32FC3);
        
        // Allocate device memory
        float *d_input, *d_kernel, *d_output;
        checkCudaError(cudaMalloc((void**)&d_input, imgSize), "Allocating d_input");
        checkCudaError(cudaMalloc((void**)&d_kernel, kernelSize), "Allocating d_kernel");
        checkCudaError(cudaMalloc((void**)&d_output, imgSize), "Allocating d_output");
        
        // Copy data to device
        checkCudaError(cudaMemcpy(d_input, imageFloat.ptr<float>(0), imgSize, cudaMemcpyHostToDevice), "Copying input");
        checkCudaError(cudaMemcpy(d_kernel, kernel.ptr<float>(0), kernelSize, cudaMemcpyHostToDevice), "Copying kernel");
        
        // Launch convolution kernel
        int total_pixels = height * width;
        int blockSize = 256;
        int numBlocks = (total_pixels + blockSize - 1) / blockSize;
        convolution_kernel<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output, height, width, channels, kernel_size);
        cudaDeviceSynchronize();
        
        // Copy result back to host
        checkCudaError(cudaMemcpy(output.ptr<float>(0), d_output, imgSize, cudaMemcpyDeviceToHost), "Copying output");
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        
        return output;
    }
    
    // Gaussian Blur: Constructs a Gaussian kernel and applies it.
    cv::Mat gaussianBlur(const cv::Mat& image, float sigma = 1.0f) {
        int kernel_size = std::max(3, (int)(2 * sigma) * 2 + 1);
        cv::Mat kernel(kernel_size, kernel_size, CV_32F);
        int center = kernel_size / 2;
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                float x = i - center;
                float y = j - center;
                float value = expf(-(x*x + y*y) / (2 * sigma * sigma));
                kernel.at<float>(i, j) = value;
                sum += value;
            }
        }
        kernel /= sum;
        return applyKernel(image, kernel);
    }
    
    // Sobel Edge Detection: Uses the convolution kernel with Sobel filters.
    cv::Mat sobelEdgeDetection(const cv::Mat& image) {
        // Define Sobel kernels (for each channel)
        cv::Mat sobelX = (cv::Mat_<float>(3,3) << -1, 0, 1,
                                                   -2, 0, 2,
                                                   -1, 0, 1);
        cv::Mat sobelY = (cv::Mat_<float>(3,3) << -1, -2, -1,
                                                    0,  0,  0,
                                                    1,  2,  1);
        cv::Mat gradX = applyKernel(image, sobelX);
        cv::Mat gradY = applyKernel(image, sobelY);
        
        cv::Mat edges(image.size(), CV_32FC3);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                cv::Vec3f gx = gradX.at<cv::Vec3f>(i, j);
                cv::Vec3f gy = gradY.at<cv::Vec3f>(i, j);
                cv::Vec3f magnitude;
                for (int c = 0; c < 3; c++) {
                    magnitude[c] = sqrtf(gx[c]*gx[c] + gy[c]*gy[c]);
                }
                edges.at<cv::Vec3f>(i, j) = magnitude;
            }
        }
        // Normalize edges to [0,1]
        double minVal, maxVal;
        cv::minMaxLoc(edges.reshape(1), &minVal, &maxVal);
        edges = edges / maxVal;
        return edges;
    }
    
    // Sharpen: Uses a fixed sharpening kernel.
    cv::Mat sharpen(const cv::Mat& image, float strength = 1.0f) {
        cv::Mat kernel = (cv::Mat_<float>(3,3) << 0, -1, 0,
                                                   -1, 5, -1,
                                                   0, -1, 0);
        return applyKernel(image, kernel);
    }
    
    // Resize: Uses a CUDA kernel with bilinear interpolation.
    cv::Mat resize(const cv::Mat& image, int new_height, int new_width) {
        int in_height = image.rows;
        int in_width = image.cols;
        int channels = image.channels();
        size_t inSize = in_height * in_width * channels * sizeof(float);
        size_t outSize = new_height * new_width * channels * sizeof(float);
        
        cv::Mat imageFloat;
        image.convertTo(imageFloat, CV_32FC3);
        
        cv::Mat output(new_height, new_width, CV_32FC3);
        
        float *d_input, *d_output;
        checkCudaError(cudaMalloc((void**)&d_input, inSize), "Allocating d_input for resize");
        checkCudaError(cudaMalloc((void**)&d_output, outSize), "Allocating d_output for resize");
        
        checkCudaError(cudaMemcpy(d_input, imageFloat.ptr<float>(0), inSize, cudaMemcpyHostToDevice), "Copying input for resize");
        
        int total_pixels = new_height * new_width;
        int blockSize = 256;
        int numBlocks = (total_pixels + blockSize - 1) / blockSize;
        resize_kernel<<<numBlocks, blockSize>>>(d_input, d_output, in_height, in_width, channels, new_height, new_width);
        cudaDeviceSynchronize();
        
        checkCudaError(cudaMemcpy(output.ptr<float>(0), d_output, outSize, cudaMemcpyDeviceToHost), "Copying output for resize");
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return output;
    }
};

// ---------------------------------------------------------------------------
// Main: Demonstrates usage and benchmarks the operations.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_image>" << endl;
        return -1;
    }
    
    CudaImageProcessor processor;
    cv::Mat img = processor.loadImage(argv[1]);
    cout << "Image loaded: " << img.cols << " x " << img.rows << endl;
    
    clock_t start, end;
    double cpu_time;
    
    // Gaussian Blur
    start = clock();
    cv::Mat blurred = processor.gaussianBlur(img, 2.0f);
    end = clock();
    cpu_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Gaussian blur completed in " << cpu_time << " seconds" << endl;
    processor.saveImage(blurred, "blurred.jpg");
    
    // Sobel Edge Detection
    start = clock();
    cv::Mat edges = processor.sobelEdgeDetection(img);
    end = clock();
    cpu_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Edge detection completed in " << cpu_time << " seconds" << endl;
    processor.saveImage(edges, "edges.jpg");
    
    // Sharpen
    start = clock();
    cv::Mat sharpened = processor.sharpen(img, 1.5f);
    end = clock();
    cpu_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Sharpening completed in " << cpu_time << " seconds" << endl;
    processor.saveImage(sharpened, "sharpened.jpg");
    
    // Resize
    start = clock();
    cv::Mat resized = processor.resize(img, 512, 512);
    end = clock();
    cpu_time = double(end - start) / CLOCKS_PER_SEC;
    cout << "Resizing completed in " << cpu_time << " seconds" << endl;
    processor.saveImage(resized, "resized.jpg");
    
    cout << "Results saved!" << endl;
    return 0;
}
