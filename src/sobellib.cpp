#include "sobellib.h"

#include <cstdint>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <cstdlib>

#define THREAD_NUM 8

#include "immintrin.h"

// function to note time
unsigned long long rdtsc() {
    unsigned long long int x;
    unsigned a, d;

    __asm__ volatile("rdtsc"
                     : "=a"(a), "=d"(d));

    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

// Function to load an image from a given path
bool loadImage(std::string inputPath, cv::Mat &inputImage) {
    // Load the image from the specified path
    inputImage = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    // Otherwise, return false
    return !inputImage.empty();
}

// Function to convert cv::Mat to 16bit signed array
void convertMatTo16BitSArray(const cv::Mat &mat, short *arr) {
    // Ensures the input Mat is of 16-bit signed type
    cv::Mat convertedMat;
    if (mat.type() != CV_16S) {
        mat.convertTo(convertedMat, CV_16S);
    } else {
        convertedMat = mat;
    }
    // Check if the Mat is continuous
    if (convertedMat.isContinuous()) {
        // If continuous, copy in one go
        std::memcpy(arr, convertedMat.data, convertedMat.total() * convertedMat.elemSize());
    } else {
        // If not, copy row by row
        for (int i = 0; i < convertedMat.rows; ++i) {
            std::memcpy(arr + i * convertedMat.cols, convertedMat.ptr<short>(i), convertedMat.cols * sizeof(short));
        }
    }
}

// Function which does Sobel filtering using OpenCV
FnOutput sobel_opencv(const cv::Mat &inputImage) {
    cv::Mat grad_x, grad_y, outputImage;
    cv::Mat abs_grad_x, abs_grad_y;
    int ddepth = CV_16S;
    int kSize = 3;
    unsigned long long t0, t1;

    // start timer
    t0 = rdtsc();
    // Sobel Gradient X
    cv::Sobel(inputImage, grad_x, ddepth, 1, 0, kSize, 1, 0, cv::BORDER_DEFAULT);
    // Sobel Gradient Y
    cv::Sobel(inputImage, grad_y, ddepth, 0, 1, kSize, 1, 0, cv::BORDER_DEFAULT);
    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x, 1, 0);
    cv::convertScaleAbs(grad_y, abs_grad_y, 1, 0);
    /// Approximate Total Gradient
    cv::addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, outputImage);
    // end timer
    t1 = rdtsc();

    FnOutput out(t1 - t0, outputImage);
    return out;
}

// Function which gaussian 3 x 3 smoothing using OpenCV
FnOutput gaussianBlur3x3_opencv(const cv::Mat &inputImage) {
    cv::Mat outputImage;
    int kSize = 3;
    unsigned long long t0, t1;

    // start timer
    t0 = rdtsc();
    // Gaussian filter X
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kSize, kSize), 0);
    // stop timer
    t1 = rdtsc();

    FnOutput out(t1 - t0, outputImage);
    return out;
}

// Function to perform Sobel X filtering using SIMD
// The function takes an input image and a pointer to the output buffer where the result will be stored
unsigned long long sobelX_SIMD(const cv::Mat &inputImage, short *outputPointer) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int N = cols + 2;

    // padded input and output
    short *inputPointer = new short[(rows + 2) * (cols + 2)];

    // initialize output to all zeros
    for (int i = 0; i < (rows + 2) * (cols + 2); ++i) {
        outputPointer[i] = 0;
    }

    // initialize padded input and convert to 16bit signed int
    cv::Mat inputPaddedImage(inputImage.rows + 2, inputImage.cols + 2, inputImage.type(), cv::Scalar(0));
    inputImage.copyTo(inputPaddedImage(cv::Rect(1, 1, inputImage.cols, inputImage.rows)));
    convertMatTo16BitSArray(inputPaddedImage, inputPointer);

    __m256i i1, i2, i3, i4, i5, i6, i7, i8, o1, o2, o3, o4, o5, o6;
    __m256i zeros = _mm256_setzero_si256();

    unsigned long long t0, t1;
    t0 = rdtsc();

    // #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 1; i <= rows - 6; i += 6) {
        for (int j = 1; j <= (cols + 2) - 17; j += 16) {
            // input loads
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j - 1));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j - 1));
            i1 = _mm256_sub_epi16(zeros, i1);
            i2 = _mm256_sub_epi16(zeros, i2);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j - 1));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j - 1));
            i3 = _mm256_sub_epi16(zeros, i3);
            i4 = _mm256_sub_epi16(zeros, i4);

            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j - 1));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j - 1));
            i5 = _mm256_sub_epi16(zeros, i5);
            i6 = _mm256_sub_epi16(zeros, i6);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j - 1));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j - 1));
            i7 = _mm256_sub_epi16(zeros, i7);
            i8 = _mm256_sub_epi16(zeros, i8);
        
            // output loads
            o1 = _mm256_lddqu_si256((__m256i *)(outputPointer + i * N + j));
            o2 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 1) * N + j));
            o1 = _mm256_add_epi16(i1, o1);
            o2 = _mm256_add_epi16(i2, o2);

            o3 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 2) * N + j));
            o4 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 3) * N + j));
            o3 = _mm256_add_epi16(i3, o3);
            o4 = _mm256_add_epi16(i4, o4);

            o5 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 4) * N + j));
            o6 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 5) * N + j));
            o5 = _mm256_add_epi16(i5, o5);
            o6 = _mm256_add_epi16(i6, o6);

            // off diagonally above computes
            o1 = _mm256_add_epi16(i3, o1);
            o2 = _mm256_add_epi16(i4, o2);
            o3 = _mm256_add_epi16(i5, o3);
            o4 = _mm256_add_epi16(i6, o4);
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            // -2 * input
            i2 = _mm256_add_epi16(i2, i2);
            i3 = _mm256_add_epi16(i3, i3);
            i4 = _mm256_add_epi16(i4, i4);
            i5 = _mm256_add_epi16(i5, i5);
            i6 = _mm256_add_epi16(i6, i6);
            i7 = _mm256_add_epi16(i7, i7);

            // same level computes
            o1 = _mm256_add_epi16(i2, o1);
            o2 = _mm256_add_epi16(i3, o2);
            o3 = _mm256_add_epi16(i4, o3);
            o4 = _mm256_add_epi16(i5, o4);
            o5 = _mm256_add_epi16(i6, o5);
            o6 = _mm256_add_epi16(i7, o6);

            // change inputs
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j + 1));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j + 1));
            o1 = _mm256_add_epi16(i1, o1);
            o2 = _mm256_add_epi16(i2, o2);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j + 1));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j + 1));
            o3 = _mm256_add_epi16(i3, o3);
            o4 = _mm256_add_epi16(i4, o4);
            
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j + 1));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j + 1));
            o5 = _mm256_add_epi16(i5, o5);
            o6 = _mm256_add_epi16(i6, o6);
            o1 = _mm256_add_epi16(i3, o1);
            o2 = _mm256_add_epi16(i4, o2);
            o3 = _mm256_add_epi16(i5, o3);
            o4 = _mm256_add_epi16(i6, o4);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j + 1));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j + 1));
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            i2 = _mm256_add_epi16(i2, i2);
            i3 = _mm256_add_epi16(i3, i3);
            i4 = _mm256_add_epi16(i4, i4);
            i5 = _mm256_add_epi16(i5, i5);
            i6 = _mm256_add_epi16(i6, i6);
            i7 = _mm256_add_epi16(i7, i7);

            o1 = _mm256_add_epi16(i2, o1);
            o2 = _mm256_add_epi16(i3, o2);
            o3 = _mm256_add_epi16(i4, o3);
            o4 = _mm256_add_epi16(i5, o4);
            o5 = _mm256_add_epi16(i6, o5);
            o6 = _mm256_add_epi16(i7, o6);

            // |Gx|
            o1 = _mm256_abs_epi16(o1);
            o2 = _mm256_abs_epi16(o2);
            o3 = _mm256_abs_epi16(o3);
            o4 = _mm256_abs_epi16(o4);
            o5 = _mm256_abs_epi16(o5);
            o6 = _mm256_abs_epi16(o6);

            // 6 stores - 6 x 16 kernel = 96 Outputs
            _mm256_storeu_si256((__m256i *)(outputPointer + i * N + j), o1);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 1) * N + j), o2);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 2) * N + j), o3);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 3) * N + j), o4);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 4) * N + j), o5);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 5) * N + j), o6);
        }
    }
    t1 = rdtsc();
    free(inputPointer);
    return (t1 - t0);
}

// Function to perform Sobel Y filtering using SIMD
// The function takes an input image and a pointer to the output buffer where the result will be stored
unsigned long long sobelY_SIMD(const cv::Mat &inputImage, short *outputPointer) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int N = cols + 2;

    // padded input and output
    short *inputPointer = new short[(rows + 2) * (cols + 2)];

    // initialize output to all zeros
    for (int i = 0; i < (rows + 2) * (cols + 2); ++i) {
        outputPointer[i] = 0;
    }

    // initialize padded input and convert to 16bit signed int
    cv::Mat inputPaddedImage(inputImage.rows + 2, inputImage.cols + 2, inputImage.type(), cv::Scalar(0));
    inputImage.copyTo(inputPaddedImage(cv::Rect(1, 1, inputImage.cols, inputImage.rows)));
    convertMatTo16BitSArray(inputPaddedImage, inputPointer);

    __m256i i1, i2, i3, i4, i5, i6, i7, i8, o1, o2, o3, o4, o5, o6;
    __m256i zeros = _mm256_setzero_si256();

    unsigned long long t0, t1;
    t0 = rdtsc();
    // #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 1; i <= rows - 6; i += 6) {
        for (int j = 1; j <= (cols + 2) - 17; j += 16) {
            // inputs
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j - 1));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j - 1));
            i1 = _mm256_sub_epi16(zeros, i1);
            i2 = _mm256_sub_epi16(zeros, i2);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j - 1));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j - 1));
            i3 = _mm256_sub_epi16(zeros, i3);
            i4 = _mm256_sub_epi16(zeros, i4);

            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j - 1));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j - 1));
            i5 = _mm256_sub_epi16(zeros, i5);
            i6 = _mm256_sub_epi16(zeros, i6);

            // output loads
            o1 = _mm256_lddqu_si256((__m256i *)(outputPointer + i * N + j));
            o2 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 1) * N + j));
            o1 = _mm256_add_epi16(i1, o1);
            o2 = _mm256_add_epi16(i2, o2);

            o3 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 2) * N + j));
            o4 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 3) * N + j));
            o3 = _mm256_add_epi16(i3, o3);
            o4 = _mm256_add_epi16(i4, o4);

            o5 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 4) * N + j));
            o6 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 5) * N + j));
            o5 = _mm256_add_epi16(i5, o5);
            o6 = _mm256_add_epi16(i6, o6);

            i3 = _mm256_sub_epi16(zeros, i3);
            i4 = _mm256_sub_epi16(zeros, i4);
            i5 = _mm256_sub_epi16(zeros, i5);
            i6 = _mm256_sub_epi16(zeros, i6);

            o1 = _mm256_add_epi16(i3, o1);
            o2 = _mm256_add_epi16(i4, o2);
            o3 = _mm256_add_epi16(i5, o3);
            o4 = _mm256_add_epi16(i6, o4);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j - 1));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j - 1));
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            // change inputs
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j));
            i1 = _mm256_add_epi16(i1, i1);
            i2 = _mm256_add_epi16(i2, i2);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j));
            i3 = _mm256_add_epi16(i3, i3);
            i4 = _mm256_add_epi16(i4, i4);

            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j));
            i5 = _mm256_add_epi16(i5, i5);
            i6 = _mm256_add_epi16(i6, i6);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j));
            i7 = _mm256_add_epi16(i7, i7);
            i8 = _mm256_add_epi16(i8, i8);
            
            o1 = _mm256_add_epi16(i3, o1);
            o2 = _mm256_add_epi16(i4, o2);
            o3 = _mm256_add_epi16(i5, o3);
            o4 = _mm256_add_epi16(i6, o4);
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            i1 = _mm256_sub_epi16(zeros, i1);
            i2 = _mm256_sub_epi16(zeros, i2);
            i3 = _mm256_sub_epi16(zeros, i3);
            i4 = _mm256_sub_epi16(zeros, i4);
            i5 = _mm256_sub_epi16(zeros, i5);
            i6 = _mm256_sub_epi16(zeros, i6);

            o1 = _mm256_add_epi16(i1, o1);
            o2 = _mm256_add_epi16(i2, o2);
            o3 = _mm256_add_epi16(i3, o3);
            o4 = _mm256_add_epi16(i4, o4);
            o5 = _mm256_add_epi16(i5, o5);
            o6 = _mm256_add_epi16(i6, o6);

            
            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j + 1));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j + 1));
            o1 = _mm256_add_epi16(i3, o1);
            o2 = _mm256_add_epi16(i4, o2);
            i3 = _mm256_sub_epi16(zeros, i3);
            i4 = _mm256_sub_epi16(zeros, i4);

            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j + 1));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j + 1));
            o3 = _mm256_add_epi16(i5, o3);
            o4 = _mm256_add_epi16(i6, o4);
            i5 = _mm256_sub_epi16(zeros, i5);
            i6 = _mm256_sub_epi16(zeros, i6);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j + 1));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j + 1));
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j + 1));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j + 1));
            i1 = _mm256_sub_epi16(zeros, i1);
            i2 = _mm256_sub_epi16(zeros, i2);

            o1 = _mm256_add_epi16(i1, o1);
            o2 = _mm256_add_epi16(i2, o2);
            o3 = _mm256_add_epi16(i3, o3);
            o4 = _mm256_add_epi16(i4, o4);
            o5 = _mm256_add_epi16(i5, o5);
            o6 = _mm256_add_epi16(i6, o6);

            // |Gy|
            o1 = _mm256_abs_epi16(o1);
            o2 = _mm256_abs_epi16(o2);
            o3 = _mm256_abs_epi16(o3);
            o4 = _mm256_abs_epi16(o4);
            o5 = _mm256_abs_epi16(o5);
            o6 = _mm256_abs_epi16(o6);

            // 6 stores = 6 x 16 kernel = 96 outputs
            _mm256_storeu_si256((__m256i *)(outputPointer + i * N + j), o1);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 1) * N + j), o2);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 2) * N + j), o3);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 3) * N + j), o4);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 4) * N + j), o5);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 5) * N + j), o6);
        }
    }
    t1 = rdtsc();
    free(inputPointer);
    return (t1 - t0);
}

// Function which computes total gradient by summing up results from SIMD Sobel X and Y
// The function takes pointers to the Sobel X, Y output and an output buffer where the result will be stored
unsigned long long approxTotalGradient_SIMD(const short *outputXPointer, const short *outputYPointer, const int size, short *outputPtr) {
    // initialize output to all zeros
    for (int i = 0; i < size; ++i) {
        outputPtr[i] = 0;
    }

    int kernelSize = 192;
    int numIterations = size / kernelSize; 

    __m256i o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12;
    __m256i i1, i2;

    unsigned long long t0, t1;

    t0 = rdtsc();
    // #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < numIterations; ++i) {
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 0));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 0));
        o1 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 16));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 16));
        o2 = _mm256_add_epi16(i1, i2);

        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 32));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 32));
        o3 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 48));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 48));
        o4 = _mm256_add_epi16(i1, i2);

        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 64));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 64));
        o5 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 80));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 80));
        o6 = _mm256_add_epi16(i1, i2);

        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 96));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 96));
        o7 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 112));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 112));
        o8 = _mm256_add_epi16(i1, i2);

        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 128));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 128));
        o9 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 144));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 144));
        o10 = _mm256_add_epi16(i1, i2);

        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 160));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 160));
        o11 = _mm256_add_epi16(i1, i2);
        i1 = _mm256_lddqu_si256((__m256i *)(outputXPointer + i * kernelSize + 176));
        i2 = _mm256_lddqu_si256((__m256i *)(outputYPointer + i * kernelSize + 176));
        o12 = _mm256_add_epi16(i1, i2);

        // 12 - stores = 12 x 16 kernel = 192 outputs
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 0), o1);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 16), o2);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 32), o3);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 48), o4);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 64), o5);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 80), o6);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 96), o7);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 112), o8);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 128), o9);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 144), o10);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 160), o11);
        _mm256_storeu_si256((__m256i *)(outputPtr + i * kernelSize + 176), o12);
    }
    t1 = rdtsc();
    return (t1 - t0);
}

// Function to perform 3 x 3 gaussian smoothing using SIMD
unsigned long long gaussianBlur3x3_SIMD(const cv::Mat &inputImage, short *outputPointer) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int N = cols + 2;

    // padded input and output
    short *inputPointer = new short[(rows + 2) * (cols + 2)];

    // initialize output to all zeros
    for (int i = 0; i < (rows + 2) * (cols + 2); ++i) {
        outputPointer[i] = 0;
    }

    // initialize padded input and convert to 16bit signed int
    cv::Mat inputPaddedImage(inputImage.rows + 2, inputImage.cols + 2, inputImage.type(), cv::Scalar(0));
    inputImage.copyTo(inputPaddedImage(cv::Rect(1, 1, inputImage.cols, inputImage.rows)));
    convertMatTo16BitSArray(inputPaddedImage, inputPointer);

    __m256i i1, i2, i3, i4, i5, i6, i7, i8;
    __m256i o1, o2, o3, o4, o5, o6;

    unsigned long long t0, t1;

    t0 = rdtsc();
    // #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 1; i <= rows - 6; i += 6) {
        for (int j = 1; j <= (cols + 2) - 16; j += 16) {
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j - 1));
            i2 = _mm256_srli_epi16 (i2, 3);
            o1 = _mm256_lddqu_si256((__m256i *)(outputPointer + i * N + j));

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j - 1));
            i3 = _mm256_srli_epi16 (i3, 3);
            o2 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 1) * N + j));

            o1 = _mm256_add_epi16(i2, o1);
            o2 = _mm256_add_epi16(i3, o2);

            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j - 1));
            i4 = _mm256_srli_epi16 (i4, 3);
            o3 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 2) * N + j));
            
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j - 1));
            i5 = _mm256_srli_epi16 (i5, 3);
            o4 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 3) * N + j));
            o3 = _mm256_add_epi16(i4, o3);
            o4 = _mm256_add_epi16(i5, o4);

            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j - 1));
            i6 = _mm256_srli_epi16 (i6, 3);
            o5 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 4) * N + j));
            
            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j - 1));
            i7 = _mm256_srli_epi16 (i7, 3);
            o6 = _mm256_lddqu_si256((__m256i *)(outputPointer + (i + 5) * N + j));
            
            o5 = _mm256_add_epi16(i6, o5);
            o6 = _mm256_add_epi16(i7, o6);

            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j - 1));
            i1 = _mm256_srli_epi16(i1, 4);            
            o1 = _mm256_add_epi16(i1, o1);

            i2 = _mm256_srli_epi16(i2, 1);
            o2 = _mm256_add_epi16(i2, o2);
            
            i3 = _mm256_srli_epi16(i3, 1);
            o3 = _mm256_add_epi16(i3, o3);
            o1 = _mm256_add_epi16(i3, o1);

            i4 = _mm256_srli_epi16(i4, 1);
            o4 = _mm256_add_epi16(i4, o4);
            o2 = _mm256_add_epi16(i4, o2);

            i5 = _mm256_srli_epi16(i5, 1);
            o5 = _mm256_add_epi16(i5, o5);
            o3 = _mm256_add_epi16(i5, o3);

            i6 = _mm256_srli_epi16(i6, 1);
            o6 = _mm256_add_epi16(i6, o6);
            o4 = _mm256_add_epi16(i6, o4);

            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j - 1));
            i8 = _mm256_srli_epi16(i8, 4);
            i7 = _mm256_srli_epi16(i7, 1);
            o6 = _mm256_add_epi16(i8, o6);
            o5 = _mm256_add_epi16(i7, o5);

            // change inputs
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j));
            i2 = _mm256_srli_epi16(i2, 2);
            o1 = _mm256_add_epi16(i2, o1);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j));
            i3 = _mm256_srli_epi16(i3, 2);
            o2 = _mm256_add_epi16(i3, o2);

            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j));
            i4 = _mm256_srli_epi16(i4, 2);
            o3 = _mm256_add_epi16(i4, o3);

            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j));
            i5 = _mm256_srli_epi16(i5, 2);
            o4 = _mm256_add_epi16(i5, o4);

            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j));
            i6 = _mm256_srli_epi16(i6, 2);
            o5 = _mm256_add_epi16(i6, o5);

            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j));
            i7 = _mm256_srli_epi16(i7, 2);
            o6 = _mm256_add_epi16(i7, o6);

            i2 = _mm256_srli_epi16(i2, 1);
            o2 = _mm256_add_epi16(i2, o2);

            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j));
            i8 = _mm256_srli_epi16(i8, 3);

            i3 = _mm256_srli_epi16(i3, 1);
            o3 = _mm256_add_epi16(i3, o3);
            o1 = _mm256_add_epi16(i3, o1);
            
            i4 = _mm256_srli_epi16(i4, 1);
            o4 = _mm256_add_epi16(i4, o4);
            o2 = _mm256_add_epi16(i4, o2);
            
            i5 = _mm256_srli_epi16(i5, 1);
            o5 = _mm256_add_epi16(i5, o5);
            o3 = _mm256_add_epi16(i5, o3);
            
            i6 = _mm256_srli_epi16(i6, 1);
            o6 = _mm256_add_epi16(i6, o6);
            o4 = _mm256_add_epi16(i6, o4);
            
            i7 = _mm256_srli_epi16(i7, 1);
            o5 = _mm256_add_epi16(i7, o5);
            
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j));
            i1 = _mm256_srli_epi16(i1, 3);
            o1 = _mm256_add_epi16(i1, o1);
            o6 = _mm256_add_epi16(i8, o6);

            // change inputs
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + i * N + j + 1));
            i2 = _mm256_srli_epi16 (i2, 3);
            o1 = _mm256_add_epi16(i2, o1);

            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 1) * N + j + 1));
            i3 = _mm256_srli_epi16 (i3, 3);
            o2 = _mm256_add_epi16(i3, o2);
            
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 2) * N + j + 1));
            i4 = _mm256_srli_epi16 (i4, 3);
            o3 = _mm256_add_epi16(i4, o3);
            
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 3) * N + j + 1));
            i5 = _mm256_srli_epi16 (i5, 3);
            o4 = _mm256_add_epi16(i5, o4);
            
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 4) * N + j + 1));
            i6 = _mm256_srli_epi16 (i6, 3);
            o5 = _mm256_add_epi16(i6, o5);
            
            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 5) * N + j + 1));
            i7 = _mm256_srli_epi16 (i7, 3);
            o6 = _mm256_add_epi16(i7, o6);
            
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i - 1) * N + j + 1));
            i1 = _mm256_srli_epi16 (i1, 4);
            o1 = _mm256_add_epi16(i1, o1);

            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i + 6) * N + j + 1));
            i8 = _mm256_srli_epi16 (i8, 4);
            
            i2 = _mm256_srli_epi16 (i2, 1);
            o2 = _mm256_add_epi16(i2, o2);

            i3 = _mm256_srli_epi16 (i3, 1);
            o3 = _mm256_add_epi16(i3, o3);
            o1 = _mm256_add_epi16(i3, o1);

            i4 = _mm256_srli_epi16 (i4, 1);
            o4 = _mm256_add_epi16(i4, o4);
            o2 = _mm256_add_epi16(i4, o2);
            
            i5 = _mm256_srli_epi16 (i5, 1);
            o5 = _mm256_add_epi16(i5, o5);
            o3 = _mm256_add_epi16(i5, o3);
            
            i6 = _mm256_srli_epi16 (i6, 1);
            o6 = _mm256_add_epi16(i6, o6);
            o4 = _mm256_add_epi16(i6, o4);

            i7 = _mm256_srli_epi16 (i7, 1);
            o5 = _mm256_add_epi16(i7, o5);
            o6 = _mm256_add_epi16(i8, o6);

            // 6 stores = 6 x 16 kernel = 96 Outputs
            _mm256_storeu_si256((__m256i *)(outputPointer + i * N + j), o1);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 1) * N + j), o2);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 2) * N + j), o3);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 3) * N + j), o4);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 4) * N + j), o5);
            _mm256_storeu_si256((__m256i *)(outputPointer + (i + 5) * N + j), o6);
        }
    }
    t1 = rdtsc();
    return (t1 - t0);
}

/** FUTURE Improvement **/
/**
void pack(const short *inputPointer, const int oRows, const int oCols, short *packedInputPointer) {
    int start = 0;
    __m256i i1, i2, i3, i4, i5, i6, i7, i8;

    for (int i = 0; i < oRows - 6; i += 6) {
        for (int j = 0; j < oCols - 18; j += 16) {
            // 8 loads and stores
            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i * oCols) + j));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 1) * oCols) + j));
            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 2) * oCols) + j));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 3) * oCols) + j));
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 4) * oCols) + j));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 5) * oCols) + j));
            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 6) * oCols) + j));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 7) * oCols) + j));

            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start), i1);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 16), i2);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 32), i3);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 48), i4);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 64), i5);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 80), i6);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 96), i7);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 112), i8);

            start += 1;

            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i * oCols) + j + 1));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 1) * oCols) + j + 1));
            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 2) * oCols) + j + 1));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 3) * oCols) + j + 1));
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 4) * oCols) + j + 1));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 5) * oCols) + j + 1));
            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 6) * oCols) + j + 1));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 7) * oCols) + j + 1));

            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start), i1);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 16), i2);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 32), i3);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 48), i4);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 64), i5);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 80), i6);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 96), i7);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 112), i8);

            start += 1;

            i1 = _mm256_lddqu_si256((__m256i *)(inputPointer + (i * oCols) + j + 2));
            i2 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 1) * oCols) + j + 2));
            i3 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 2) * oCols) + j + 2));
            i4 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 3) * oCols) + j + 2));
            i5 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 4) * oCols) + j + 2));
            i6 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 5) * oCols) + j + 2));
            i7 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 6) * oCols) + j + 2));
            i8 = _mm256_lddqu_si256((__m256i *)(inputPointer + ((i + 7) * oCols) + j + 2));

            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start), i1);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 16), i2);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 32), i3);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 48), i4);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 64), i5);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 80), i6);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 96), i7);
            _mm256_storeu_si256((__m256i *)(packedInputPointer + 128 * start + 112), i8);

            start += 1;
        }
    }
}
*/
/** FUTURE Improvement **/
