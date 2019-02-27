#ifndef LOSS_H
#define LOSS_H

#include <vector>

struct SquareLoss
{
	static float loss(float *yout, float *yexpected, unsigned n)
	{
		float totalLoss = 0;

		for(unsigned i = 0; i < n; ++i)
		{
			float e = yout[i] - yexpected[i];
			totalLoss += e*e;
		}

		return totalLoss;
	}

	static std::vector<float> derivative(float *yout, float *yexpected, unsigned n)
	{
		std::vector<float> result;
		result.reserve(n);

		for(unsigned i = 0; i < n; ++i)
			result.push_back(yout[i] - yexpected[i]);

		return result;
	}
};

#endif