#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <exception>
#include <functional>

template<typename T>
T takeFirst(std::vector<T> &v)
{
	T r = v.at(0);
	v.erase(v.begin());
	return r;
}

struct Node
{
	virtual void forward() = 0;

	// public since they need to be accessible by the graph class
	std::vector<Node*> parents;
	std::vector<Node*> children;

	bool executed = false;
	bool derivated = false;

	float getResult(int i) { return outputs.at(i); }
	float getDerivative(int index) { return derivatives.at(index); }
	float getDerivative(Node* n)
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
	
	void computeDerivatives()
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

	bool isReadyForward()
	{
		bool ready = true;
		for(auto p : parents)
		{
			if(!p->executed)
				ready = false;	
		}

		return ready;
	}
	
	bool isReadyBackward()
	{
		bool ready = true;
		for(auto c : children)
		{
			if(!c->derivated)
				ready = false;	
		}

		return ready;
	}

protected:
	std::vector<float> outputs;
	std::vector<float> derivatives;
	std::vector<float> partialDerivatives;
};

struct InputNode : public Node
{
	InputNode() { outputs.push_back(0); }
	explicit InputNode (float i) { outputs.push_back(i); }
	void setInput(float i) { outputs.at(0) = i; }
	float getInput() { return outputs.at(0); }

	virtual void forward() { partialDerivatives = {1}; }
};

bool nodeReadyFwd(Node* a, Node* b) { return a->isReadyForward() && !b->isReadyForward(); }
bool nodeReadyBwd(Node* a, Node* b) { return a->isReadyBackward() && !b->isReadyBackward(); }

struct Graph
{
	std::vector<InputNode*> inputNodes;
	std::vector<Node*> outputNodes;

	void setInputs(const std::vector<float> &inputs)
	{
		unsigned s = std::min(inputs.size(), inputNodes.size());
		for(unsigned i = 0; i < s; ++i)
		{
			inputNodes.at(i)->setInput(inputs.at(i));
		}
	}

	float getOutput(int i=0, int j=0)
	{
		return outputNodes.at(i)->getResult(j);
	}

	void traverse()
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

	void backProp(int index)
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

	void traverseNodes( std::function<void(Node*)> visit )
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

	void setGraphUnexecuted()
	{
		traverseNodes(
			[](Node* n) 
			{
				n->executed = false;
			}
		);
	}

	void setGraphUnderivated()
	{
		traverseNodes(
			[](Node* n) 
			{
				n->derivated = false;
			}
		);		
	}
};

struct AdditionNode : public Node
{
	AdditionNode(Node* a, Node* b, int indexA=0, int indexB=0)
	: iA(indexA), iB(indexB)
	{
		parents.push_back(a);
		parents.push_back(b);

		a->children.push_back(this);
		b->children.push_back(this);

		outputs.push_back(0);
	}

	void forward() 
	{ 
		outputs.at(0) = parents.at(0)->getResult(iA) + parents.at(1)->getResult(iB);
		partialDerivatives = {1, 1};
	}

private:
	int iA;
	int iB;
};

struct MultiplicationNode : public Node
{
	MultiplicationNode(Node* a, Node* b, int indexA=0, int indexB=0)
	: iA(indexA), iB(indexB)
	{
		parents.push_back(a);
		parents.push_back(b);

		a->children.push_back(this);
		b->children.push_back(this);

		outputs.push_back(0);
	}

	void forward() 
	{
		float x = parents.at(0)->getResult(iA);
		float y = parents.at(1)->getResult(iB);
		
		outputs.at(0) = x * y;
		partialDerivatives = {y, x};
	}

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
	SigmoidNode(Node* p, int index=0)
	: pIndex(index)
	{
		parents.push_back(p);
		p->children.push_back(this);
		outputs.push_back(0);
	}

	void forward() 
	{
		float x = parents.at(0)->getResult(pIndex);
		float z = 1.0f / (1.0f + exp(-1.0f*x));
		
		outputs.at(0) = z;
		partialDerivatives = { z*(1.0f-z) };
	}

private:
	int pIndex;
};

void addMultTest(float x, float y)
{
	Graph graph;
	
	InputNode a, b;
	AdditionNode c(&a, &b);
	MultiplicationNode o(&a, &c);	

	// graph computes
	// o = a * (a + b)

	graph.inputNodes = {&a, &b};
	graph.outputNodes = {&o};

	graph.setInputs({x, y});
	graph.traverse();

	float actual = graph.getOutput(0);
	float expected= x*(x+y);

	if(actual != expected) { std::cout << "Test equal failed. Expected: " << x << ". Got: " << y << "." << std::endl; }
}

int main()
{
	addMultTest(3, 11);
	addMultTest(1, 2);
	addMultTest(4, 5);

	// if(1) return 0;

	Graph graph;
	
	InputNode x1, x2;
	InputNode w1, w2;

	MultiplicationNode m1(&x1, &w1);
	MultiplicationNode m2(&x2, &w2);

	AdditionNode a(&m1, &m2);

	SigmoidNode s(&a);

	graph.inputNodes = {&x1, &x2, &w1, &w2};
	graph.setInputs({     1,   1, 0,   0});
	graph.outputNodes = {&s};

	float goalOutput = 0.6;
	float learningRate = 0.01;

	// let's try and train this graph
	for(int i = 0; i < 100; ++i)
	{
		graph.traverse();
		float output = graph.getOutput(0);
		std::cout << "Computation result: " << output << std::endl;
		
		graph.backProp(0);
		
		{
			float w = w1.getInput();
			float doutput_dw = w1.getDerivative(0);
			float sign = output > goalOutput ? 1 : -1;
			w -= sign * doutput_dw * learningRate;
			w1.setInput(w);			
		}

		{
			float w = w2.getInput();
			float doutput_dw = w2.getDerivative(0);
			float sign = output > goalOutput ? 1 : -1;
			w -= sign * doutput_dw * learningRate;
			w2.setInput(w);			
		}

	}

	return 0;
}