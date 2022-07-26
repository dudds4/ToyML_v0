#include "graph.h"
#include "nodetypes.h"

#include <algorithm>
#include <iostream>
#include <queue>


template<typename T>
T takeFirst(std::vector<T> &v)
{
	T r = v.at(0);
	v.erase(v.begin());
	return r;
}

// Node implementations

double Node::getOutput() { return output; }
double Node::getDerivative(int index) { return derivatives.at(index); }
double Node::getDerivative(Node* n)
{
	auto it = std::find(parents.begin(), parents.end(), n);

	if(it == parents.end())
	{
		std::cout << "getDerivative:\t couldn't find pointer in parents vector." << std::endl; 
		throw new std::exception();
	}

	unsigned i = it - parents.begin();
	return getDerivative(i);
}

void Node::computeDerivatives(double L)
{
	// if I don't have children, I'm an output node
	// otherwise, sum the child derivatives

	if(children.size())
	{
		L = 0;
		for(auto c : children)
			L += c->getDerivative(this);
	}

	unsigned nDerivs = partialDerivatives.size();
	derivatives.resize(nDerivs);
	for(int i = 0; i < nDerivs; ++i)
		derivatives.at(i) = L * partialDerivatives.at(i);
}

bool Node::isReadyForward()
{
	bool ready = true;
	for(auto p : parents)
	{
		if(!p->executed)
			ready = false;	
	}

	return ready;
}

bool Node::isReadyBackward()
{
	bool ready = true;
	for(auto c : children)
	{
		if(!c->derivated)
			ready = false;	
	}

	return ready;
}

void Node::setParent(Node* n)
{
	// remove self from parents...
	for(auto p : parents)
	{
		p->children.erase(
			std::remove(p->children.begin(), p->children.end(), this),
			p->children.end());
	}

	parents.clear();
	parents.push_back(n);
	n->children.push_back(this);

	partialDerivatives.resize(parents.size());
}

void Node::setParents(const std::vector<Node*> &parentV)
{
	// remove self from parents...
	for(auto p : parents)
	{
		p->children.erase(
			std::remove(p->children.begin(), p->children.end(), this),
			p->children.end());
	}

	parents.clear();
	parents.reserve(parentV.size());

	for(auto n : parentV)
	{
		parents.push_back(n);
		n->children.push_back(this);
	}

	partialDerivatives.resize(parents.size());	
}

bool nodeReadyFwd(Node* a, Node* b) { return a->isReadyForward() && !b->isReadyForward(); }
bool nodeReadyBwd(Node* a, Node* b) { return a->isReadyBackward() && !b->isReadyBackward(); }

std::vector<double> Graph::forwardPass(const std::vector<double> &inputValues)
{
	std::vector<double> result;

	setInputs(inputValues);
	traverse();

	for(auto n : outputNodes)
		result.push_back(n->getOutput());
	
	return result;
}

std::vector<double> Graph::forwardPass(const double* inputValues)
{
	std::vector<double> result;

	setInputs(inputValues, inputNodes.size());
	traverse();

	for(auto n : outputNodes)
		result.push_back(n->getOutput());
	
	return result;
}


void Graph::addInputNodes(const std::vector<InputNode*> &inputs)
{
	inputNodes.insert(inputNodes.end(), inputs.begin(), inputs.end());
}

void Graph::addParamNodes(const std::vector<InputNode*> &params)
{
	paramNodes.reserve(paramNodes.size() + params.size());
	for(auto n : params)
	{
		if(!n)
			throw new std::exception();

		paramNodes.push_back(n);
	}
}

void copyValuesToInputNodes(const std::vector<double> &values, std::vector<InputNode*> &nodes)
{
	unsigned s = std::min(nodes.size(), values.size());
	for(unsigned i = 0; i < s; ++i)
	{
		nodes.at(i)->setInput(values.at(i));
	}	
}

void copyValuesToInputNodes(const double* values, unsigned n, std::vector<InputNode*> &nodes)
{
	if(n != nodes.size())
		throw new std::exception();

	for(unsigned i = 0; i < n; ++i)
		nodes.at(i)->setInput(values[i]);
}

void Graph::setInputs(const std::vector<double> &values)
{
	copyValuesToInputNodes(values, inputNodes);
}

void Graph::setInputs(const double* values, unsigned n)
{
	copyValuesToInputNodes(values, n, inputNodes);
}

void Graph::setParams(const std::vector<double> &values)
{
	copyValuesToInputNodes(values, paramNodes);
}


void Graph::updateParams(std::function<double(double,double)> update )
{
	for(auto node : paramNodes)
	{
		double w = node->getInput();
		double deriv = node->getDerivative(0);
		node->setInput(update(w, deriv));
	}
}

double Graph::getOutput(int i)
{
	return outputNodes.at(i)->getOutput();
}

void Graph::traverse()
{
	setGraphUnexecuted();

	std::vector<Node*> q;
	for(auto n : inputNodes)
		q.push_back(n);

	for(auto n : paramNodes)
		q.push_back(n);

	while(q.size())
	{
		auto n = takeFirst(q);

		if(!n->executed)
		{
			if(n->isReadyForward())
			{
				n->forward();
				n->executed = true;
				for(auto c : n->children)
					q.push_back(c);
			}
			else
			{
				// pretty sure this is unreachable with current architecture & declarative flow
				// as such its kind of untested, but in theory it should work
				// std::cout << "traverse had to sort!" << std::endl;

				// move the ready nodes to the front of the queue
				std::sort(q.begin(), q.end(), nodeReadyFwd);

				// add our node back into the queue
				q.push_back(n);
			}			
		}

	}
}


void Graph::backProp(const double *baseDeriv, unsigned n)
{
	if(n != outputNodes.size())
		throw new std::exception();


	setGraphUnderivated();	
	std::vector<Node*> q;

	unsigned i = 0;
	// do the output layer explicitly
	for(auto oNode : outputNodes)
	{
		oNode->computeDerivatives(baseDeriv[i++]);
		oNode->derivated = true;
		for(auto p : oNode->parents)
			q.push_back(p);		
	}

	// q.push_back(outputNodes.at(index));

	while(q.size())
	{
		auto n = takeFirst(q);

		if(n->isReadyBackward())
		{
			n->computeDerivatives();
			n->derivated = true;
			for(auto p : n->parents)
				q.push_back(p);
		}
		else
		{
			// pretty sure this is unreachable with current architecture & declarative flow
			// as such its kind of untested, but in theory it should work

			// std::cout << "backprop had to sort!" << std::endl;
			// move the ready nodes to the front of the queue
			std::sort(q.begin(), q.end(), nodeReadyBwd);
			// add our node back into the queue
			q.push_back(n);
		}
	}
}

void Graph::backProp(const std::vector<double>& baseDeriv)
{
	backProp(baseDeriv.data(), baseDeriv.size());
}

// void Graph::backProp(int index, float baseDeriv)
// {
// 	if(index > outputNodes.size()) { std::cout << "backProp out of index exception\n"; throw new std::exception(); }

// 	setGraphUnderivated();	
// 	std::vector<Node*> q;

// 	// do the first explicitly
// 	auto oNode = outputNodes.at(index);
// 	oNode->computeDerivatives(baseDeriv);
// 	oNode->derivated = true;
// 	for(auto p : oNode->parents)
// 		q.push_back(p);

// 	// q.push_back(outputNodes.at(index));

// 	while(q.size())
// 	{
// 		auto n = takeFirst(q);

// 		if(n->isReadyBackward())
// 		{
// 			n->computeDerivatives();
// 			n->derivated = true;
// 			for(auto p : n->parents)
// 				q.push_back(p);
// 		}
// 		else
// 		{
// 			// pretty sure this is unreachable with current architecture & declarative flow
// 			// as such its kind of untested, but in theory it should work

// 			std::cout << "backprop had to sort!" << std::endl;
// 			// move the ready nodes to the front of the queue
// 			std::sort(q.begin(), q.end(), nodeReadyBwd);
// 			// add our node back into the queue
// 			q.push_back(n);
// 		}
// 	}
// }

void Graph::traverseNodes( std::function<void(Node*)> visit )
{
	std::queue<Node*> q;
	
	for(auto n : inputNodes)
		q.push(n);

	for(auto n : paramNodes)
		q.push(n);

	while(q.size())
	{
		auto n = q.front();
		q.pop();

		visit(n);

		for(auto c : n->children)
			q.push(c);
	}		
}

void Graph::setGraphUnexecuted()
{
	traverseNodes(
		[](Node* n) 
		{
			n->executed = false;
		}
	);
}

void Graph::setGraphUnderivated()
{
	traverseNodes(
		[](Node* n) 
		{
			n->derivated = false;
		}
	);		
}

