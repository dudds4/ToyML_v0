#include "graph.h"
#include "nodetypes.h"

#include <iostream>
#include <cstdlib>

void additionTest(float x, float y);
void multiplicationTest(float x, float y);
void addMultTest(float x, float y);
void vectorMultTest();

int main()
{
	srand(time(NULL));

	// all the tests
	additionTest(rand(), rand());
	multiplicationTest(rand(), rand());
	addMultTest(rand(), rand());
	vectorMultTest();

	return 0;
}

#define ASSERT_EQUAL(a, b) do { if((a) != (b)) std::cout << "Test equal failed. Expected: " << (a) << ". Got: " << (b) << "." << std::endl; } while(0)

void additionTest(float x, float y)
{
	Graph graph;
	
	InputNode a, b;
	AdditionNode o(&a, &b);	

	// graph computes
	// o = (a + b)

	graph.inputNodes = {&a, &b};
	graph.outputNodes = {&o};

	graph.setInputs({x, y});
	graph.traverse();

	float actual = graph.getOutput(0);
	float expected= x + y;

	ASSERT_EQUAL(actual, expected);
}

void multiplicationTest(float x, float y)
{
	Graph graph;
	
	InputNode a, b;
	MultiplicationNode o(&a, &b);

	// graph computes
	// o = a * b

	graph.inputNodes = {&a, &b};
	graph.outputNodes = {&o};

	graph.setInputs({x, y});
	graph.traverse();

	float actual = graph.getOutput(0);
	float expected= x*y;

	ASSERT_EQUAL(actual, expected);
}

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

	ASSERT_EQUAL(actual, expected);
}

void vectorMultTest()
{
	const int L = 4;
	
	std::vector<Node*> inputs;
	std::vector<Node*> weights;
	std::vector<InputNode*> v;

	InputNode w[L];
	InputNode x[L];

	float expected = 0;
	for(int i = 0; i < L; ++i)
	{
		inputs.push_back(&x[i]);
		weights.push_back(&w[i]);
		
		v.push_back(&x[i]);
		v.push_back(&w[i]);

		w[i].setInput(rand());
		x[i].setInput(rand());
		expected += w[i].getInput() * x[i].getInput();
	}

	VectorMultNode vmNode(inputs, weights);

	Graph graph;
	
	graph.inputNodes = v;
	graph.outputNodes = {&vmNode};

	graph.traverse();

	float actual = graph.getOutput(0);

	ASSERT_EQUAL(actual, expected);
}