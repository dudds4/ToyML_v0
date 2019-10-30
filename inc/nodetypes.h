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

// struct VectorInputNode : public Node
// {
// 	VectorInputNode(int size);
// 	explicit VectorInputNode (int size);
// 	void setInput(const std::vector<float>& src);
// 	void getInput(std::vector<float>& dst);
// };

struct AdditionNode : public Node
{
	AdditionNode(Node* a, Node* b);
	virtual void forward();
};

struct MultiplicationNode : public Node
{
	MultiplicationNode() {}
	MultiplicationNode(Node* a, Node* b);
	virtual void forward();
};

struct VectorMultNode : public Node
{
	VectorMultNode();
	VectorMultNode(std::vector<Node*> inputs, std::vector<Node*> weights);
	virtual void forward();
	void setInputs(std::vector<Node*> inputs, std::vector<Node*> weights);
};

struct SigmoidNode : public Node
{
	SigmoidNode();
	SigmoidNode(Node* p);
	virtual void forward();
};

template <typename ForwardFunc, typename BackwardFunc>
struct FunctionNode : public Node
{
	FunctionNode() {}
	FunctionNode(Node* p) { setParent(p); }

	virtual void forward()
	{
		auto p = parents.at(0);
		float in = p->getOutput();
		output = ForwardFunc(in);

		partialDerivatives.at(0) = BackwardFunc(in);
	}
};

struct MaxNode : public Node
{
	MaxNode() {}
	MaxNode(const std::vector<Node*>& p);
	virtual void forward();
};

struct InverseNode : public Node
{
	InverseNode() {}
	InverseNode(Node* p) { setParent(p); }

	virtual void forward();
};

#endif // NODETYPES_H
