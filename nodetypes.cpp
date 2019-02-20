#include "nodetypes.h"
#include <cmath>

// ---------------------- Input Node ----------------------

InputNode::InputNode() { outputs.push_back(0); }
InputNode::InputNode (float i) { outputs.push_back(i); }

void InputNode::setInput(float i) { outputs.at(0) = i; }
float InputNode::getInput() { return outputs.at(0); }
void InputNode::forward() { partialDerivatives = {1}; }

// ---------------------- Addition Node ----------------------

AdditionNode::AdditionNode(Node* a, Node* b, int indexA, int indexB)
: iA(indexA), iB(indexB)
{
	parents.push_back(a);
	parents.push_back(b);

	a->children.push_back(this);
	b->children.push_back(this);

	outputs.push_back(0);
}

void AdditionNode::forward() 
{ 
	outputs.at(0) = parents.at(0)->getResult(iA) + parents.at(1)->getResult(iB);
	partialDerivatives = {1, 1};
}

// ---------------------- Multiplication Node ----------------------

MultiplicationNode::MultiplicationNode(Node* a, Node* b, int indexA, int indexB)
: iA(indexA), iB(indexB)
{
	parents.push_back(a);
	parents.push_back(b);

	a->children.push_back(this);
	b->children.push_back(this);

	outputs.push_back(0);
}

void MultiplicationNode::forward() 
{
	float x = parents.at(0)->getResult(iA);
	float y = parents.at(1)->getResult(iB);
	
	outputs.at(0) = x * y;
	partialDerivatives = {y, x};
}

// ---------------------- Sigmoid Node ----------------------

SigmoidNode::SigmoidNode(Node* p, int index)
: pIndex(index)
{
	parents.push_back(p);
	p->children.push_back(this);
	outputs.push_back(0);
}

void SigmoidNode::forward() 
{
	float x = parents.at(0)->getResult(pIndex);
	float z = 1.0f / (1.0f + exp(-1.0f*x));
	
	outputs.at(0) = z;
	partialDerivatives = { z*(1.0f-z) };
}