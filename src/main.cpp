#include <iostream>
#include <cstdlib>

#include "sobellib.h"

#define MAX_FREQ 3.2
#define BASE_FREQ 2.4

int main(int argc, char **argv) {
    // define number of runs and input image path
    int runs = 0;
    std::string inputImagePath;

    if (argc < 3) {
        std::cout << "sobel_simd <Number of Runs> <Input Image Path>" << std::endl;
        return EXIT_FAILURE;
    }

    runs = atoi(argv[1]);
    inputImagePath = argv[2];

    // initialize variables to count cycles
    unsigned long long sum1 = 0;
    unsigned long long sum2 = 0;
    unsigned long long sum3 = 0;
    unsigned long long sum4 = 0;
    unsigned long long sum5 = 0;
    unsigned long long sum6 = 0;

    // load input test image
    cv::Mat inputImage;

    if (!loadImage(inputImagePath, inputImage)) {
        std::cout << "Error loading input image" << std::endl;
        return EXIT_FAILURE;
    }

    int rows = inputImage.rows;
    int cols = inputImage.cols;

    std::cout << "Processing " << rows << " rows" << std::endl;
    std::cout << "Runs " << runs << std::endl;

    short *gaussianBlurOutputPointer = new short[(rows + 2) * (cols + 2)];
    short *sobelX_outputPointer = new short[(rows + 2) * (cols + 2)];
    short *sobelY_outputPointer = new short[(rows + 2) * (cols + 2)];
    short *outputPtr = new short[(rows + 2) * (cols + 2)];

    // for defined number of runs run
    // sobel naive, opencv and simd version
    for (int r = 0; r != runs; ++r) {
        // benchmarking sobel opencv
        sum1 += sobel_opencv(inputImage).time;
        // simd sobel simd x v2
        sum2 += sobelX_SIMD(inputImage, sobelX_outputPointer);
        // simd sobel simd y v2
        sum3 += sobelY_SIMD(inputImage, sobelY_outputPointer);
        // simd total gradient
        sum4 += approxTotalGradient_SIMD(sobelX_outputPointer, sobelY_outputPointer, (rows + 2) * (cols + 2), outputPtr);
        // simd gaussian blur
        sum5 += gaussianBlur3x3_SIMD(inputImage, gaussianBlurOutputPointer);
        // opencv gaussian blur
        sum6 += gaussianBlur3x3_opencv(inputImage).time;
    }

    std::cout << "[Sobel OpenCV] Number of cycles: " << ((double)sum1 * MAX_FREQ / BASE_FREQ) / (runs) << std::endl;
    std::cout << "[Gaussian Blur OpenCV] Number of cycles: " << ((double)sum6 * MAX_FREQ / BASE_FREQ) / (runs) << std::endl;

    std::cout << "[Sobel SIMD] Number of cycles: " << ((double)(sum2 + sum3 + sum4) * MAX_FREQ / BASE_FREQ) / (runs) << std::endl;
    std::cout << "[Gaussian Blur 3 x 3 SIMD] Number of cycles: " << ((double)sum5 * MAX_FREQ / BASE_FREQ) / (runs) << std::endl;

    std::cout << "[Sobel X SIMD] throughput: " << ((double)((rows + 2) * (cols + 2) * runs)) / ((double)sum2 * MAX_FREQ / BASE_FREQ) << std::endl;
    std::cout << "[Sobel Y SIMD] throughput: " << ((double)((rows + 2) * (cols + 2) * runs)) / ((double)sum3 * MAX_FREQ / BASE_FREQ) << std::endl;
    std::cout << "[Gaussian SIMD] throughput: " << ((double)((rows + 2) * (cols + 2) * runs)) / ((double)sum5 * MAX_FREQ / BASE_FREQ) << std::endl;

    FnOutput opencv_sobel_out = sobel_opencv(inputImage);
    FnOutput opencv_gaussian_out = gaussianBlur3x3_opencv(inputImage);

    cv::Mat sobelXMat(rows + 2, cols + 2, CV_16S, sobelX_outputPointer);
    cv::Mat sobelYMat(rows + 2, cols + 2, CV_16S, sobelY_outputPointer);
    cv::Mat outputMat(rows + 2, cols + 2, CV_16S, outputPtr);
    cv::Mat gaussianMat(rows + 2, cols + 2, CV_16S, gaussianBlurOutputPointer);
    
    std::cout << "Saving input image reference" << std::endl;
    cv::imwrite("../out/input_ref.png", inputImage);

    std::cout << "Saving OpenCV sobel filter result" << std::endl;
    cv::imwrite("../out/sobel_opencv.png", opencv_sobel_out.filteredImage);
    
    std::cout << "Saving OpenCV gaussian filter result" << std::endl;
    cv::imwrite("../out/gaussian_opencv.png", opencv_gaussian_out.filteredImage);

    std::cout << "Saving gaussian result" << std::endl;
    cv::imwrite("../out/gaussian_simd.png", gaussianMat);

    std::cout << "Saving SIMD x sobel filter result" << std::endl;
    cv::imwrite("../out/sobelx_simd.png", sobelXMat);

    std::cout << "Saving SIMD y sobel filter result" << std::endl;
    cv::imwrite("../out/sobely_simd.png", sobelYMat);

    std::cout << "Saving final result" << std::endl;
    cv::imwrite("../out/sobel_simd.png", outputMat);

    free(gaussianBlurOutputPointer);
    free(sobelX_outputPointer);
    free(sobelY_outputPointer);
    free(outputPtr);
    return 0;
}
