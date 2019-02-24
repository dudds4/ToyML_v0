#ifndef LOSS_H
#define LOSS_H

struct SquareLoss
{
	static float loss(float yout, float yexpected)
	{
		float x = yout - yexpected;
		return x * x;
	}

	static float derivative(float yout, float yexpected)
	{
		return 2 * (yout - yexpected);
	}
};

#endif