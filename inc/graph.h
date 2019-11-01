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

	double getOutput();
	double getDerivative(int index);
	double getDerivative(Node* n);
	void computeDerivatives(double downstream=1);
	bool isReadyForward();
	bool isReadyBackward();

	void setParent(Node* n);
	void setParents(const std::vector<Node*> &parentV);

protected:
	double output = 0;
	std::vector<double> derivatives;
	std::vector<double> partialDerivatives;
};

struct InputNode;

struct Graph
{
	std::vector<InputNode*> inputNodes;
	std::vector<InputNode*> paramNodes;
	std::vector<Node*> outputNodes;

	void addInputNodes(const std::vector<InputNode*> &inputs);
	void addParamNodes(const std::vector<InputNode*> &inputs);

	void setInputs(const std::vector<double> &values);
	void setInputs(const double* values, unsigned n);

	void setParams(const std::vector<double> &values);
	void updateParams(std::function<double(double,double)> update );

	std::vector<double> forwardPass(const std::vector<double> &inputValues);
	std::vector<double> forwardPass(const double* inputValues);

	double getOutput(int i=0);
	void traverse();

	void backProp(const double *baseDeriv, unsigned n);
	void backProp(const std::vector<double>& baseDeriv);
	
	void traverseNodes( std::function<void(Node*)> visit );
	void setGraphUnexecuted();
	void setGraphUnderivated();
};

#endif//GRAPH_H