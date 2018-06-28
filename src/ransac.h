#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <utility>

using namespace std;

struct PointDelta {
	glm::vec2 delta;  // the delta vector
	float dist;  // distance between the delta vector and a reference vector
	int origPos;
};

class RansacSeparator {
	private:
		bool * devPointGroup;  // length: N
		glm::vec2 * devPointDiffs;  // length: N
		PointDelta * devPointDeltas;  // length: N
		thrust::device_ptr<PointDelta> thrust_devPointDeltas;
	public:
		int N;  // number of (edge) points
		void computeDiffs(PointDelta & tempDelta, glm::vec2 & meanVector, int THRESHOLD_N, int ITERATIONS, float SCALE_THRESHOLD_N);
		RansacSeparator(int N);
		~RansacSeparator();

    	pair<glm::vec2, glm::vec2> separate(bool * pointGroup, glm::vec2 * pointDiffs, float THRESHOLD, int ITERATIONS);
};
