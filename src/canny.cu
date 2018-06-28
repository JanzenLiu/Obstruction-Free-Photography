#include "canny.h"
#include "timing.h"
#include <cstdio>
#include <cmath>

#define UPPERTHRESHOLD 90
#define LOWERTHRESHOLD 30

// Sobel Operator for gradient calculation
const float G_x[3 * 3] = {
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1
};

// Sobel Operator for gradient calculation
const float G_y[3 * 3] = {
  1, 2, 1,
  0, 0, 0,
  -1, -2, -1
};

// Gaussian Kernel for smoothing (noise removal)
const float gaussian[5 * 5] = {
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  5.f/159, 12.f/159, 15.f/159, 12.f/159, 2.f/159,
  4.f/159, 9.f/159, 12.f/159, 9.f/159, 4.f/159,
  2.f/159, 4.f/159, 5.f/159, 4.f/159, 2.f/159
};


/**
 * Smooth a given image using a given kernel.
 *
 * @param N         Number of pixels of the input image, supposed to be (width * height)
 * @param width     Width of the input image
 * @param height    Height of the input image
 * @param in        The input image as a char array of length N
 * @param out       Char array to store the smoothed image
 * @param kernel    Kernel used to smooth the input image
 * @param kernSize  Size of the kernel, i.e. height (width) of the kernel
 * */
__global__ void kernSmooth(int N, int width, int height, unsigned char * in, unsigned char * out,
                           const float * kernel, int kernSize) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
    float c = 0.0f;
	for (int i = 0; i < kernSize; i++) {
		int tx = x + i - kernSize / 2;
		for (int j = 0; j < kernSize; j++) {
			int ty = y + j - kernSize / 2;
			if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
				c += in[ty * width + tx] * kernel[j * kernSize + i];
			}
		}
	}
	out[y * width + x] = fabs(c);
}


/**
 * Calculate intensity gradient (magnitude and direction) from a given image using given kernels.
 *
 * @param N         Number of pixels of the input image, supposed to be (width * height)
 * @param width     Width of the input image
 * @param height    Height of the input image
 * @param in        The input image as a char array of length N
 * @param gradient  The gradient magnitudes of the input image, as a char array of length N
 * @param edgeDir   The gradient directions of the input image, as a char array of lenght N
 * @param G_x       Kernel used to compute x-component of the gradient
 * @param G_y       Kernel used to compute y-component of the gradient
 * */
__global__ void kernGradient(int N, int width, int height, unsigned char * in, unsigned char * gradient,
                             unsigned char * edgeDir, float * G_x, float * G_y) {
    // TODO: to verify, should gradient be float?
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}

	// ===============================
	// x- and y- Component Calculation
    // ===============================
    int idx, dx, dy, tx, ty;
    float Gx, Gy, grad, angle;
    idx = y * width + x;
    Gx = Gy = 0;
    for (dy = 0; dy < 3; dy++) {  // TODO: to abstract 3 to a G_y size parameter
        ty = y + dy - 1;
        for (dx = 0; dx < 3; dx++) {  // TODO: to abstract 3 to a G_x size parameter
            tx = x + dx - 1;
            if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
                Gx += in[ty * width + tx] * G_x[dy * 3 + dx];
                Gy += in[ty * width + tx] * G_y[dy * 3 + dx];
            }
        }
    }

    // ===================================
    // Magnitude and Direction Calculation
    // ===================================
    grad = sqrt(Gx * Gx + Gy * Gy);
    angle = (atan2(Gx, Gy) / 3.14159f) * 180.0f;
    unsigned char roundedAngle;
    if (((-22.5 < angle) && (angle <= 22.5)) || ((157.5 < angle) && (angle <= -157.5))) {
        roundedAngle = 0;
    }
    if (((-157.5 < angle) && (angle <= -112.5)) || ((22.5 < angle) && (angle <= 67.5))) {
        roundedAngle = 45;
    }
	if (((-112.5 < angle) && (angle <= -67.5)) || ((67.5 < angle) && (angle <= 112.5))) {
        roundedAngle = 90;
    }
	if (((-67.5 < angle) && (angle <= -22.5)) || ((112.5 < angle) && (angle <= 157.5))) {
        roundedAngle = 135;
    }
    gradient[idx] = grad;
    edgeDir[idx] = roundedAngle;
}


/**
 * Compress (Turn off) non-maximum gradient in a given gradient map.
 *
 * @param N         Number of pixels of the original image (gradient map), supposed to be (width * height)
 * @param width     Width of the original image (gradient map)
 * @param height    Height of the original image (gradient map)
 * @param in        The gradient direction map as a char array of length N
 * @param out       The gradient magnitude map as a char array of length N
 * */
__global__ void nonMaxSuppression(int N, int width, int height, unsigned char * in, unsigned char * out) {
    int D = 1;
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= width || y >= height) {
		return;
	}
    int angle = in[y * width + x];
    switch(angle) {
        case 0:
            if (out[y * width + x] < out[(y + D) * width + x] || out[y * width + x] < out[(y - D) * width + x]) {
                out[y * width + x] = 0;
            }
            break;
        case 45:
            if (out[y * width + x] < out[(y + D) * width + x - D] || out[y * width + x] < out[(y - D) * width + x + D]) {
                out[y * width + x] = 0;
            }
            break;
        case 90:
            if (out[y * width + x] < out[y * width + x + D] || out[y * width + x] < out[y * width + x - D]) {
                out[y * width + x] = 0;
            }
            break;
        case 135:
            if (out[y * width + x] < out[(y + D) * width + x + D] || out[y * width + x] < out[(y - D) * width + x - D]) {
                out[y * width + x] = 0;
            }
            break;
        default:
            break;
    }
}


/**
 * Perform hystersis thresholding to a given gradient map.
 *
 * @param N         Number of pixels of the original image (gradient map), supposed to be (width * height)
 * @param width     Width of the original image (gradient map)
 * @param height    Height of the original image (gradient map)
 * @param in        The gradient magnitude map as a char array of length N
 * */
__global__ void hysteresis(int N, int width, int height, unsigned char * in) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}
    int idx = y * width + x;
    if (in[idx] > UPPERTHRESHOLD) {
        in[idx] = 255;
    } else if (in[idx] < LOWERTHRESHOLD) {
        in[idx] = 0;
    } else {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nidx = (y + dy) * width + (x + dx);
                if(0 <= (y + dy) && (y + dy) < height &&  0 <= (x + dx) && (x + dx) < width && in[nidx] > LOWERTHRESHOLD) {  // TODO: to verify
                    in[nidx] = 255;
                }
            }
        }
    }
}


/**
 * Get edge map from a
 *
 * @param N         Number of pixels of the input image, supposed to be (width * height)
 * @param width     Width of the input image
 * @param height    Height of the input image
 * @param in        The input image as a char array, whose length is supposed to be N, i.e. (width * height)
 * @return          Calculated edge map, as a char array of length N,
 *                  where 255 indicates an edge pixel, and 0 indicates a non-edge pixel
 * */
unsigned char * Canny::edge(int N, int width, int height, unsigned char * in) {
	unsigned char * dev_in, * dev_gradient;
    unsigned char * smooth, * gradient, * edgeDir;
    float * blur_kernel, * gradient_x, * gradient_y;

    // =================================
    // Memory Allocation & Value Copying
    // =================================
    gradient = new unsigned char[N];
    cudaMalloc(&smooth, sizeof(unsigned char) * N);  // to store the smoothed image
    cudaMalloc(&dev_in, sizeof(unsigned char) * N);
    cudaMalloc(&dev_gradient, sizeof(unsigned char) * N);
    cudaMalloc(&edgeDir, sizeof(unsigned char) * N);
    cudaMalloc(&blur_kernel, sizeof(float) * 5 * 5);
    cudaMalloc(&gradient_x, sizeof(float) * 3 * 3);
    cudaMalloc(&gradient_y, sizeof(float) * 3 * 3);

    cudaMemcpy(dev_in, in, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blur_kernel, gaussian, sizeof(float) * 5 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(gradient_x, G_x, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(gradient_y, G_y, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);

    // =====================
    // GPU Parallelism Setup
    // =====================
    const dim3 blockSize2d(8,8);  // threads in block
    const dim3 blocksPerGrid2d((width + blockSize2d.x - 1) / blockSize2d.x,
                               (height + blockSize2d.y - 1) / blockSize2d.y);  // blocks in grid

    // ==============
    // Core Steps:
    //  1. Smoothing
    //  2. Gradient Calculation
    //  3. Non-maximum Edge Suppression
    //  4. Hysteresis
    // ==============
    TIMEINIT
    TIMEIT((kernSmooth<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, dev_in, smooth, blur_kernel, 5)), "Smoothing")
    TIMEIT((kernGradient<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, smooth, dev_gradient, edgeDir, gradient_x, gradient_y)), "Gradient")
    TIMEIT((nonMaxSuppression<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, edgeDir, dev_gradient)), "Non-maximum suppression")
    TIMEIT((hysteresis<<<blocksPerGrid2d, blockSize2d>>>(N, width, height, dev_gradient)), "Hysteresis") // can use stream compaction
    TIMEEND

    cudaMemcpy(gradient, dev_gradient, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(smooth);
    cudaFree(edgeDir);
    cudaFree(blur_kernel);
    cudaFree(gradient_x);
    cudaFree(gradient_y);
	cudaFree(dev_gradient);

    return gradient;
}
