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

float Node::getResult(int i) { return outputs.at(i); }
float Node::getDerivative(int index) { return derivatives.at(index); }
float Node::getDerivative(Node* n)
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

void Node::computeDerivatives()
{
	float L = 1;

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

bool nodeReadyFwd(Node* a, Node* b) { return a->isReadyForward() && !b->isReadyForward(); }
bool nodeReadyBwd(Node* a, Node* b) { return a->isReadyBackward() && !b->isReadyBackward(); }

void Graph::setInputs(const std::vector<float> &inputs)
{
	unsigned s = std::min(inputs.size(), inputNodes.size());
	for(unsigned i = 0; i < s; ++i)
	{
		inputNodes.at(i)->setInput(inputs.at(i));
	}
}

float Graph::getOutput(int i, int j)
{
	return outputNodes.at(i)->getResult(j);
}

void Graph::traverse()
{
	setGraphUnexecuted();

	std::vector<Node*> q;
	for(auto n : inputNodes)
		q.push_back(n);

	while(q.size())
	{
		auto n = takeFirst(q);

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
			std::cout << "traverse had to sort!" << std::endl;

			// move the ready nodes to the front of the queue
			std::sort(q.begin(), q.end(), nodeReadyFwd);

			// add our node back into the queue
			q.push_back(n);
		}
	}
}

void Graph::backProp(int index)
{
	if(index > outputNodes.size()) { std::cout << "backProp out of index exception\n"; throw new std::exception(); }

	setGraphUnderivated();	
	std::vector<Node*> q;
	q.push_back(outputNodes.at(index));

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

			std::cout << "backprop had to sort!" << std::endl;
			// move the ready nodes to the front of the queue
			std::sort(q.begin(), q.end(), nodeReadyBwd);
			// add our node back into the queue
			q.push_back(n);
		}
	}
}

void Graph::traverseNodes( std::function<void(Node*)> visit )
{
	std::queue<Node*> q;
	for(auto n : inputNodes)
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