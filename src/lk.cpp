#include <time.h>
#include "timing.h"
#include "lk.h"

/**
 * Calculate sparse optical flow between two given images for edge points only
 *
 * @param N         Number of pixels of either image, supposed to be (width * height)
 * @param width     Width of either image
 * @param height    Height of either image
 * @param grad      The gradient magnitude map for the reference image
 * @param img1      The reference image as a char array, whose length is supposed to be N, i.e. (width * height)
 * @param img2      The moved image as a char array, whose length is supposed to be N, i.e. (width * height)
 * @return          Edge flow as a pair of vectors of points
 * */
pair<vector<Point2f>, vector<Point2f>> lucasKanade(int N, int width, int height, unsigned char * grad,
                                                   unsigned char * img1, unsigned char * img2) {
    vector<Point2f> points[2];  // points[0]: point positions in the original image, points[1]: ... in the moved image
    Mat mat1(Size(width, height), CV_8UC1, img1);
    Mat mat2(Size(width, height), CV_8UC1, img2);

    // ========================
    // Determine Feature Points
    // ========================
    // i.e. edge points in this case
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grad[y * width + x] == 255) {
                points[0].push_back(Point2f(x, y));
            }
        }
    }

    // ==============================
    // Find Positions of Moved Points
    // ==============================
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(mat1, mat2, points[0], points[1], status, err, Size(21, 21));

    // ======================
    // Calculate Sparse Flows
    // ======================
    for (int i = 0; i < points[1].size(); i++) {
        if (status[i] == 0) continue;
        points[1][i] -= points[0][i];
    }

    return make_pair(points[0], points[1]);
}

/**
 * Calculate sparse optical flow between two given images for edge points only
 *
 * @param N         Number of pixels of either image, supposed to be (width * height)
 * @param width     Width of either image
 * @param height    Height of either image
 * @param grad      The gradient magnitude map for the reference image
 * @param img1      The reference image as a char array, whose length is supposed to be N, i.e. (width * height)
 * @param img2      The moved image as a char array, whose length is supposed to be N, i.e. (width * height)
 * @return          Edge flow as a pair of vectors, the first one is a vector of positions, and the second
 *                  one is a vector of corresponding movement/displacement
 * */
pair<vector<Point2f>, vector<Point2f>> lkEdgeFlow(int N, int width, int height, unsigned char * grad,
                                                  unsigned char * img1, unsigned char * img2) {
    pair<vector<Point2f>, vector<Point2f>> tmp;
    CPUTIMEINIT
    CPUTIMEIT(tmp = lucasKanade(N, width, height, grad, img1, img2), "Lucas Kanade")
    return tmp;  // tmp.first: vector of positions, tmp.second: vector of movement/displacement
}
