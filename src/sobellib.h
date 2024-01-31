#pragma once

#include <opencv2/opencv.hpp>

class FnOutput {
   public:
    // time it took to complete the sobel function
    unsigned long long time = 0;
    // the output filtered image after sobel
    cv::Mat filteredImage;
    // constructor
    FnOutput(unsigned long long time, cv::Mat filteredImage) {
        this->time = time;
        this->filteredImage = filteredImage;
    }
};

// Function to load an image from the specified path
bool loadImage(std::string inputPath, cv::Mat &inputImage);

// Function to apply the Sobel filter using OpenCV
FnOutput sobel_opencv(const cv::Mat &inputImage);

// Function to apply the gaussian 3 x 3 filter using OpenCV
FnOutput gaussianBlur3x3_opencv(const cv::Mat &inputImage);

// Function converts cv::Mat to 16bit signed aray
void convertMatTo16BitSArray(const cv::Mat &mat, short *arr);

// Function to apply gaussian 3 x 3 filter using SIMD
unsigned long long gaussianBlur3x3_SIMD(const cv::Mat &inputImage, short *outputPointer);

// Function to apply the Sobel x filter using SIMD
unsigned long long sobelX_SIMD(const cv::Mat &inputImage, short *outputPointer);

// Function to apply the Sobel y filter using SIMD
unsigned long long sobelY_SIMD(const cv::Mat &inputImage, short *outputPointer);

// Function to approximate total gradient
unsigned long long approxTotalGradient_SIMD(const short *outputXPointer, const short *outputYPointer, const int size, short *outputPtr);
