#include "nodetypes.h"
#include <cmath>

// ---------------------- Input Node ----------------------

InputNode::InputNode() { output=0; }
InputNode::InputNode (float i) { output = i; }

void InputNode::setInput(float i) { output = i; }
float InputNode::getInput() { return output; }
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
	output = parents.at(0)->getResult() + parents.at(1)->getResult();
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
	float x = parents.at(0)->getResult();
	float y = parents.at(1)->getResult();
	
	output = x * y;
	partialDerivatives = {y, x};
}

// ---------------------- Sigmoid Node ----------------------

SigmoidNode::SigmoidNode(Node* p)
{
	parents.push_back(p);
	p->children.push_back(this);
	output = 0;
}

void SigmoidNode::forward() 
{
	float x = parents.at(0)->getResult();
	float z = 1.0f / (1.0f + exp(-1.0f*x));
	
	output = z;
	partialDerivatives = { z*(1.0f-z) };
}