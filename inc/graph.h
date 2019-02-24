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

	float getOutput();
	float getDerivative(int index);
	float getDerivative(Node* n);
	void computeDerivatives(float downstream=1);
	bool isReadyForward();
	bool isReadyBackward();

	void setParent(Node* n);
	void setParents(const std::vector<Node*> &parentV);

protected:
	float output = 0;
	std::vector<float> derivatives;
	std::vector<float> partialDerivatives;
};

struct InputNode;

struct Graph
{
	std::vector<InputNode*> inputNodes;
	std::vector<InputNode*> paramNodes;
	std::vector<Node*> outputNodes;

	void addInputNodes(const std::vector<InputNode*> &inputs);
	void addParamNodes(const std::vector<InputNode*> &inputs);

	void setInputs(const std::vector<float> &values);
	void setParams(const std::vector<float> &values);
	void updateParams(std::function<float(float,float)> update );

	std::vector<float> forwardPass(const std::vector<float> &inputValues);

	float getOutput(int i=0);
	void traverse();
	void backProp(int index, float baseDeriv=1);
	void traverseNodes( std::function<void(Node*)> visit );
	void setGraphUnexecuted();
	void setGraphUnderivated();
};

#endif//GRAPH_H