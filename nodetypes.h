#ifndef GRAPH_NODE_TYPES_H
#define GRAPH_NODE_TYPES_H

#include "graph.h"

struct InputNode : public Node
{
	InputNode();
	explicit InputNode (float i);
	void setInput(float i);
	float getInput();

	virtual void forward();
};

struct AdditionNode : public Node
{
	AdditionNode(Node* a, Node* b);
	virtual void forward();
};

struct MultiplicationNode : public Node
{
	MultiplicationNode(Node* a, Node* b);
	virtual void forward();
};

struct VectorMultNode : public Node
{
	VectorMultNode(std::vector<Node*> inputs, std::vector<Node*> weights);
	virtual void forward();
};

struct SigmoidNode : public Node
{
	SigmoidNode(Node* p);
	virtual void forward();
};

#endif // NODETYPES_H
