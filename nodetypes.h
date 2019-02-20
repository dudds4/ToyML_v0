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
	AdditionNode(Node* a, Node* b, int indexA=0, int indexB=0);

	void forward();

private:
	int iA;
	int iB;
};

struct MultiplicationNode : public Node
{
	MultiplicationNode(Node* a, Node* b, int indexA=0, int indexB=0);
	void forward();

private:
	int iA;
	int iB;
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
	SigmoidNode(Node* p, int index=0);
	void forward();

private:
	int pIndex;
};

#endif // NODETYPES_H
