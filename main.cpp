#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <exception>
#include <functional>

#include "graph.h"
#include "nodetypes.h"

void addMultTest(float x, float y)
{
	Graph graph;
	
	InputNode a, b;
	AdditionNode c(&a, &b);
	MultiplicationNode o(&a, &c);	

	// graph computes
	// o = a * (a + b)

	graph.inputNodes = {&a, &b};
	graph.outputNodes = {&o};

	graph.setInputs({x, y});
	graph.traverse();

	float actual = graph.getOutput(0);
	float expected= x*(x+y);

	if(actual != expected) { std::cout << "Test equal failed. Expected: " << x << ". Got: " << y << "." << std::endl; }
}

int main()
{
	addMultTest(3, 11);
	addMultTest(1, 2);
	addMultTest(4, 5);

	// if(1) return 0;

	Graph graph;
	
	InputNode x1, x2;
	InputNode w1, w2;

	// MultiplicationNode m1(&x1, &w1);
	// MultiplicationNode m2(&x2, &w2);

	// AdditionNode a(&m1, &m2);

	VectorMultNode a({&x1, &x2}, {&w1, &w2});

	SigmoidNode s(&a);

	graph.inputNodes = {&x1, &x2, &w1, &w2};
	graph.setInputs({     1,   1, 0,   0});
	graph.outputNodes = {&s};

	float goalOutput = 0.6;
	float learningRate = 0.01;

	// let's try and train this graph
	for(int i = 0; i < 100; ++i)
	{
		graph.traverse();
		float output = graph.getOutput(0);
		std::cout << "Computation result: " << output << std::endl;
		
		graph.backProp(0);
		
		{
			float w = w1.getInput();
			float doutput_dw = w1.getDerivative(0);
			float sign = output > goalOutput ? 1 : -1;
			w -= sign * doutput_dw * learningRate;
			w1.setInput(w);			
		}

		{
			float w = w2.getInput();
			float doutput_dw = w2.getDerivative(0);
			float sign = output > goalOutput ? 1 : -1;
			w -= sign * doutput_dw * learningRate;
			w2.setInput(w);			
		}

	}

	return 0;
}