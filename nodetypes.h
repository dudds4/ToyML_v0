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

	void forward();
};

struct MultiplicationNode : public Node
{
	MultiplicationNode(Node* a, Node* b);
	void forward();
};

// struct MatrixMultNode : public Node
// {
// 	MatrixMultNode(std::vector<Node*> inputs, std::vector<Node*> weights)
// 	{
// 		parents.push_back(p);
// 		p->children.push_back(this);
// 	}	
// };

struct SigmoidNode : public Node
{
	SigmoidNode(Node* p);
	void forward();
};

#endif // NODETYPES_H
