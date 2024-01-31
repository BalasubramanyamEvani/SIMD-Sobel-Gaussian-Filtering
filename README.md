# Steps to Make the Code Work

This guide provides steps to get the Sobel and Gaussian implementations up and running using OpenCV

## Prerequisites

- Ensure that OpenCV is installed on the ECE cluster machine. Follow the guidelines provided in the Piazza post.
- The code has been tested with OpenCV version 3.4.20 and on Intel ®Xeon ®CPU E5-2640 v4 @ 2.40GHz (Broadwell x86 64 architecture).

## Installation and Execution

1. Change directory to `src/build`:

```sh
cd src/build
```

2. If present, delete the `CMakeCache.txt` file.

3. Run CMake from the current directory:

```sh
cmake ../
```

This step will generate the `sobel_simd` executable.

4. To use the executable, run the following command:

```sh
sobel_simd <Number of Runs> <Input Image Path>
```

This command runs OpenCV Sobel, Gaussian, and SIMD Sobel and Gaussian implementations, and prints out the number of cycles and throughput information in terms of pixels per cycle.

# Implementation Details

The implementations are located in `sobellib.cpp`, which includes:

- `gaussianBlur3x3_SIMD`: SIMD implementation of the 3 x 3 Gaussian Kernel.
- `sobelY_SIMD`: SIMD implementation of the Sobel Y kernel (Gy).
- `sobelX_SIMD`: SIMD implementation of the Sobel X kernel (Gx).
- `approxTotalGradient_SIMD`: An element-wise addition kernel, necessary for summing up the total gradients (Gx and Gy).

Please see `out` folder to see the results.

[Sobel OpenCV](out/sobel_opencv.png)
[Sobel SIMD](out/sobel_simd.png)

[Gaussian OpenCV](out/gaussian_opencv.png)
[Gaussian SIMD](out/gaussian_simd.png)

## Note

This is done as part of final project work at CMU. For the full report details, please see the report pdf file.
