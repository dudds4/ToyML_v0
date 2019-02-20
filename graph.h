#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <functional>

struct Node
{
	virtual void forward() = 0;

	// public since they need to be accessible by the graph class
	std::vector<Node*> parents;
	std::vector<Node*> children;

	bool executed = false;
	bool derivated = false;

	float getResult(int i);
	float getDerivative(int index);
	float getDerivative(Node* n);
	void computeDerivatives();
	bool isReadyForward();
	bool isReadyBackward();

protected:
	std::vector<float> outputs;
	std::vector<float> derivatives;
	std::vector<float> partialDerivatives;
};

struct InputNode;

struct Graph
{
	std::vector<InputNode*> inputNodes;
	std::vector<Node*> outputNodes;

	void setInputs(const std::vector<float> &inputs);
	float getOutput(int i=0, int j=0);
	void traverse();
	void backProp(int index);
	void traverseNodes( std::function<void(Node*)> visit );
	void setGraphUnexecuted();
	void setGraphUnderivated();
};

#endif//GRAPH_H