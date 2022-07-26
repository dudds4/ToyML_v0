#include "graph.h"
#include "nodetypes.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

void additionTest(double x, double y);
void multiplicationTest(double x, double y);
void addMultTest(double x, double y);
void vectorMultTest();
double randFloatRange(double lower, double higher);

template<typename T>
void tprint(T item)
{
	std::cout << item ;
}

template<typename T, typename ... Types>
void tprint(T item, Types ... args)
{
	std::cout << item << ' ';
	tprint(args...);
}

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

#define ABS(a) ((a) > 0 ? (a) : -(a))
#define ASSERT_EQUAL(a, b) 				do { if((a) != (b)) std::cout << "Test equal failed. Expected: " << (a) << ". Got: " << (b) << "." << std::endl; } while(0)
#define ASSERT_FLOAT_EQUAL(a,b,tol) 	do { if( ABS((a) - (b)) > tol) tprint("Test equal failed. Expected: ", a, ". Got", b, " (diff=", (b)-(a), ").\n"); } while(0)

double randFloatRange(double lower, double higher)
{
	return rand() * (higher - lower) / RAND_MAX + lower;
}

void additionTest(double x, double y)
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

	double actual = graph.getOutput(0);
	double expected = x + y;

	ASSERT_FLOAT_EQUAL(actual, expected, 1e-6);
}

void multiplicationTest(double x, double y)
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

	double actual = graph.getOutput(0);
	double expected= x*y;

	ASSERT_FLOAT_EQUAL(actual, expected, 1e-6);
}

void addMultTest(double x, double y)
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

	double actual = graph.getOutput(0);
	double expected= x*(x+y);

	ASSERT_FLOAT_EQUAL(actual, expected, 1e-6);
}

void vectorMultTest()
{
	const int L = 4;
	
	std::vector<Node*> inputs;
	std::vector<Node*> weights;
	std::vector<InputNode*> v;

	InputNode w[L];
	InputNode x[L];

	double expected = 0;
	for(int i = 0; i < L; ++i)
	{
		inputs.push_back(&x[i]);
		weights.push_back(&w[i]);
		
		v.push_back(&x[i]);
		v.push_back(&w[i]);

		w[i].setInput(randFloatRange(-100,100));
		x[i].setInput(randFloatRange(-100,100));
		expected += w[i].getInput() * x[i].getInput();
	}

	VectorMultNode vmNode(inputs, weights);

	Graph graph;
	
	graph.inputNodes = v;
	graph.outputNodes = {&vmNode};

	graph.traverse();

	double actual = graph.getOutput(0);

	ASSERT_FLOAT_EQUAL(actual, expected, 1e-6);
}