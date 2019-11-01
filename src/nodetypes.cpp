#include "nodetypes.h"
#include <cmath>

// ---------------------- Input Node ----------------------

InputNode::InputNode() { output=0; }
InputNode::InputNode (double i) { output = i; }

void InputNode::setInput(double i) { output = i; }
double InputNode::getInput() { return output; }
void InputNode::forward() { partialDerivatives = {1}; }

// ---------------------- Addition Node ----------------------

AdditionNode::AdditionNode(Node* a, Node* b)
{
	parents.push_back(a);
	parents.push_back(b);

	a->children.push_back(this);
	b->children.push_back(this);

	output = 0;
}

void AdditionNode::forward() 
{ 
	output = parents.at(0)->getOutput() + parents.at(1)->getOutput();
	partialDerivatives = {1, 1};
}

// ---------------------- Multiplication Node ----------------------

MultiplicationNode::MultiplicationNode(Node* a, Node* b)
{
	parents.push_back(a);
	parents.push_back(b);

	a->children.push_back(this);
	b->children.push_back(this);

	output = 0;
}

void MultiplicationNode::forward() 
{
	double x = parents.at(0)->getOutput();
	double y = parents.at(1)->getOutput();
	
	output = x * y;
	partialDerivatives = {y, x};
}

// ---------------------- Sigmoid Node ----------------------
SigmoidNode::SigmoidNode(){}

SigmoidNode::SigmoidNode(Node* p)
{
	parents.push_back(p);
	p->children.push_back(this);
}

void SigmoidNode::forward() 
{
	double x = parents.at(0)->getOutput();
	double z = 1.0 / (1.0 + exp(-1.0*x));
	
	output = z;
	partialDerivatives = { z*(1.0-z) };
}

// ---------------------- Vector Multiplication Node ----------------------

VectorMultNode::VectorMultNode() {}

void VectorMultNode::setInputs(std::vector<Node*> inputs, std::vector<Node*> weights)
{
	if(inputs.size() != weights.size())
	{
		throw new std::exception();
	}

	// setParents clears the previous parents vector
	// and removes this from all the parents' children vectors
	setParents(inputs);

	// then we add the weights as well
	for(auto w : weights)
	{
		parents.push_back(w);
		w->children.push_back(this);
	}

	partialDerivatives.resize(parents.size());
}

VectorMultNode::VectorMultNode(std::vector<Node*> inputs, std::vector<Node*> weights)
{
	if(inputs.size() != weights.size())
	{
		throw new std::exception();
	}

	for(auto n : inputs)
	{
		parents.push_back(n);
		n->children.push_back(this);
	}

	for(auto w : weights)
	{
		parents.push_back(w);
		w->children.push_back(this);
	}
	partialDerivatives.resize(parents.size());
}

void VectorMultNode::forward()
{
	output = 0;
	unsigned l = parents.size() / 2;

	double x,w;
	for(unsigned i = 0; i < l; ++i)
	{
		x = parents.at(i)->getOutput();
		w = parents.at(l+i)->getOutput();
		
		output += x*w;

		partialDerivatives.at(i) = w;
		partialDerivatives.at(l+i) = x;
	}
}

MaxNode::MaxNode(const std::vector<Node*>& p) { setParents(p); }

void MaxNode::forward()
{
	unsigned n = parents.size();
	double max = parents.at(0)->getOutput();
	unsigned index = 0;


	for(int i = 1; i < n; ++i)
	{
		double o = parents.at(i)->getOutput();
		if(o > max)
		{
			max = o;
			index = i;
		}
	}

	output = max;

	for(int i = 0; i < n; ++i)
	{
		partialDerivatives.at(i) = index == i ? 1 : 0;
	}
}

void InverseNode::forward()
{
	double i = parents.at(0)->getOutput();
	output = 1.0 / i;
	partialDerivatives.at(0) = -1.0 / (i*i);
}

