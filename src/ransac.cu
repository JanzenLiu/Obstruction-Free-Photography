#include "ransac.h"
#include "timing.h"

const int BLOCK_SIZE = 128;
static dim3 blocksPerGrid;  // to be assigned on construction of the RansacSeperator instance


/**
 * Compute difference between a reference vector and every PointDelta in an PointDelta array.
 *
 * @param N             Number of PointDelta
 * @param v             Reference vector as a glm::vec2
 * @param pointDeltas   Array of PointDelta to compute distance to the reference vector for
 * */
__global__ void kernComputeDiffs(int N, glm::vec2 v, PointDelta * pointDeltas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    pointDeltas[idx].dist = glm::distance(v, pointDeltas[idx].delta);
    return;
}


/**
 * Set array of PointDelta according to a given array of glm::vec2 as point differences.
 *
 * @param N             Number of points
 * @param pointDiffs    Point differences (movements) as an array of glm::vec2
 * @param pointDeltas   Array of PointDelta to hold the output
 * */
__global__ void kernGeneratePointDeltas(int N, glm::vec2 * pointDiffs, PointDelta * pointDeltas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    pointDeltas[idx].origPos = idx;
    pointDeltas[idx].delta = pointDiffs[idx];
    return;
}


/**
 * Set the ownership of the first N points of the given PointDelta array as true in the given boolean array.
 *
 * @param N             Number of points to set as true
 * @param pointDeltas   Array of PointDelta representing the N points
 * @param pointGroup    Array of boolean representing the ownership of the N points
 * */
__global__ void kernSetPointGroup(int N, PointDelta * pointDeltas, bool * pointGroup) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    pointGroup[pointDeltas[idx].origPos] = true;
    return;
}


struct SortByDist {
    __host__ __device__ bool operator()(const PointDelta & d1, const PointDelta & d2) {
        return d1.dist < d2.dist;
    }
};


struct AddDelta {
    __host__ __device__ PointDelta operator()(const PointDelta & d1, const PointDelta & d2) {
        PointDelta pd;
        pd.delta = d1.delta + d2.delta;
        return pd;
    }
};


/**
 * RansacSeparator constructor.
 *
 * @param N     Number of points to seperate
 * */
RansacSeparator::RansacSeparator(int N) {
    this->N = N;
    blocksPerGrid = dim3((this->N + BLOCK_SIZE - 1) / BLOCK_SIZE);  // static variable
    cudaMalloc(&this->devPointDeltas, N * sizeof(PointDelta));
    cudaMalloc(&this->devPointDiffs, N * sizeof(glm::vec2));
    cudaMalloc(&this->devPointGroup, N * sizeof(bool));
    this->thrust_devPointDeltas = thrust::device_pointer_cast(this->devPointDeltas);
}


/**
 * RansacSeparator destructor.
 * */
RansacSeparator::~RansacSeparator() {
    cudaFree(this->devPointDiffs);
    cudaFree(this->devPointDeltas);
    cudaFree(this->devPointGroup);
}


/**
 * Iterate RANSAC process to compute mean vector of the majority of all vectors, as well as update the distances
 *
 * @param tempDelta             PointDelta to hold some intermediate results, no need to be set as a parameter,
 *                              so just remove it in newer version (if any)
 * @param meanVector            glm::vec2 to hold the mean vector of the target vectors (i.e. those
 *                              THRESHOLD_N vectors with smallest magnitudes, the majority of the vectors)
 * @param THRESHOLD_N           Expected number of the target vectors (i.e. the majority)
 * @param ITERATIONS            Number of iterations to compute the mean vector
 * @param SCALE_THRESHOLD_N     The inverse of THRESHOLD_N, no need to be set as a parameter, so just remove it
 *                              in newer version (if any)
 * */
void RansacSeparator::computeDiffs(PointDelta & tempDelta, glm::vec2 & meanVector,
                                   int THRESHOLD_N, int ITERATIONS, float SCALE_THRESHOLD_N) {
    for (int i = 0; i < ITERATIONS; i++) {
        kernComputeDiffs<<<blocksPerGrid, BLOCK_SIZE>>>(this->N, meanVector, this->devPointDeltas);
        cudaDeviceSynchronize();
        thrust::sort(this->thrust_devPointDeltas, this->thrust_devPointDeltas + this->N, SortByDist());
        tempDelta.delta = glm::vec2(0.0f, 0.0f);
        tempDelta = thrust::reduce(this->thrust_devPointDeltas, this->thrust_devPointDeltas + THRESHOLD_N, tempDelta, AddDelta());
        meanVector = tempDelta.delta * SCALE_THRESHOLD_N;
    }
}


/**
 * Seperate a given set of vectors into two groups, and compute the center for both groups
 *
 * @param pointGroup    Array of boolean to store the ownership of the N points
 * @param pointDiffs    N difference (movement) vectors as an array of glm::vec2
 * @param THRESHOLD     Expeceted proportion of the target points (the majority)
 * @param ITERATIONS    Number of RANSAC iterations used to compute the mean vector of the target points
 * @return              a pair glm::vec2, the first one is the mean vector of the target vectors (majority),
 *                      and the second is the mean vector of the remaining vectors (minority)
 * */
pair<glm::vec2,glm::vec2> RansacSeparator::separate(bool * pointGroup, glm::vec2 * pointDiffs,
                                                    float THRESHOLD, int ITERATIONS) {
    cudaMemcpy(this->devPointDiffs, pointDiffs, this->N * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    TIMEINIT
    TIMEIT((kernGeneratePointDeltas<<<blocksPerGrid, BLOCK_SIZE>>>(this->N, this->devPointDiffs, this->devPointDeltas)), "Generating Deltas")
    cudaDeviceSynchronize();

    int THRESHOLD_N = THRESHOLD * (float)this->N;  // th * N
	printf("threshold: %d out of %d\n", THRESHOLD_N, this->N);
    float SCALE_THRESHOLD_N = 1.0f / (float)THRESHOLD_N;  // 1 / (th * N)
	float SCALE_REMAINDER = 1.0f / (float) (this->N - THRESHOLD_N);  // 1 / ((1 - th) * N)

    // ========================================
    // RANSAC Iteration to Seperate the Vectors
    // ========================================
    PointDelta tempDelta;
    glm::vec2 meanVector(0.0f, 0.0f);
    TIMEIT(computeDiffs(tempDelta, meanVector, THRESHOLD_N, ITERATIONS, SCALE_THRESHOLD_N), "Computing Diffs")

    // ===================
    // Compute Mean Vector
    // ===================
    tempDelta.delta = glm::vec2(0.0f, 0.0f);
	tempDelta = thrust::reduce(this->thrust_devPointDeltas + THRESHOLD_N, this->thrust_devPointDeltas + this->N, tempDelta, AddDelta());
	tempDelta.delta *= SCALE_REMAINDER;
	printf("%f %f, %f %f\n", meanVector.x, meanVector.y, tempDelta.delta.x, tempDelta.delta.y);

	cudaMemset(this->devPointGroup, 0, this->N * sizeof(bool));
    TIMEIT((kernSetPointGroup<<<blocksPerGrid, BLOCK_SIZE>>>(THRESHOLD_N, this->devPointDeltas, this->devPointGroup)), "Set Point Group")
    TIMEEND
	cudaDeviceSynchronize();
    cudaMemcpy(pointGroup, this->devPointGroup, this->N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
    return make_pair(meanVector, tempDelta.delta);
}
