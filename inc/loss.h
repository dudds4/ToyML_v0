#ifndef LOSS_H
#define LOSS_H

#include <vector>

struct SquareLoss
{
	static double loss(double *yout, double *yexpected, unsigned n)
	{
		double totalLoss = 0;

		for(unsigned i = 0; i < n; ++i)
		{
			double e = yout[i] - yexpected[i];
			totalLoss += e*e;
		}

		return totalLoss;
	}

	static std::vector<double> derivative(double *yout, double *yexpected, unsigned n)
	{
		std::vector<double> result;
		result.reserve(n);

		for(unsigned i = 0; i < n; ++i)
			result.push_back(2*(yout[i] - yexpected[i]));

		return result;
	}
};

#endif