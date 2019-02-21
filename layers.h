#ifndef LAYERS_H
#define LAYERS_H

#include "graph.h"
#include "nodetypes.h"

#include <vector>
#include <memory>
#include <exception>
#include <iostream>

template<typename NodeT>
struct NodeSet
{
	NodeSet(unsigned size)
	: m_size(size)
	{
		if(!size)
			throw new std::exception();

		inputs = std::shared_ptr<NodeT[]>(new NodeT[size]);
	}

	NodeSet(const std::vector<Node*> &parents)
	: m_size(parents.size())
	{
		if(!m_size)
			throw new std::exception();

		inputs = std::shared_ptr<NodeT[]>(new NodeT[m_size]);
		
		for(unsigned i = 0; i < m_size; ++i)
		{
			inputs[i].setParents({parents.at(i)});
		}
	}

	NodeT& at(size_t index)
	{
		if(index > m_size)
			throw new std::exception();

		return inputs[index];
	}

	NodeT* ptrAt(size_t index)
	{
		if(index > m_size)
			throw new std::exception();

		auto ptr = inputs.get();
		return (ptr + index);		
	}

	std::vector<Node*> getNodes()
	{
		std::vector<Node*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = 0; i < m_size; ++i)
			r.push_back(ptr+i);

		return r;
	}

	std::vector<Node*> getNodes(int l, int h)
	{
		std::vector<Node*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = l; i < m_size && i < h; ++i)
			r.push_back(ptr+i);

		return r;
	}

	std::vector<NodeT*> getInputs()
	{
		std::vector<NodeT*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = 0; i < m_size; ++i)
			r.push_back(ptr + i);

		return r;
	}

private:
	std::shared_ptr<NodeT[]> inputs;
	unsigned m_size;
};

struct LinearLayer
{
	LinearLayer(const std::vector<Node*>& inputs, size_t nOutputs)
	: m_inputs(inputs)
	, numInputs(inputs.size() + 1)
	, numOutputs(nOutputs)
	, weights((inputs.size()+1)*nOutputs)
	, bias(1)
	{
		if(!inputs.size())
			throw new std::exception();

		m_inputs.push_back(&bias);

		vectorNodes = std::shared_ptr<VectorMultNode[]>(new VectorMultNode[nOutputs]);

		unsigned cols = numInputs;
		for(unsigned r = 0; r < numOutputs; ++r)
		{
			std::vector<Node*> w;
			for(unsigned c = 0; c < cols; ++c)
				w.push_back(weights.ptrAt(r*cols + c));

			vectorNodes[r].setInputs(m_inputs, w);
		}

		// set bias.executed to always be true
		// then it doesn't need to be part of the graph
		bias.executed = true;
	}

	std::vector<InputNode*> getWeightNodes()
	{
		return weights.getInputs();
	}

	std::vector<Node*> getOutputNodes()
	{
		std::vector<Node*> result;
		for(size_t i = 0; i < numOutputs; ++i)
			result.push_back(vectorNodes.get() + i);
		
		return result;
	}

	InputNode* getBiasNode() { return &bias; }

	void setWeights(unsigned row, std::vector<float> w)
	{
		if(row > numOutputs)
			throw new std::exception();

		auto cols = numInputs;

		if(w.size() != cols)
		{
			std::cout << "size of vector w did not match number of weights in row" << std::endl;
			throw new std::exception();
		}

		for(int c = 0; c < cols; c++)
			weights.at(row*cols + c).setInput(w.at(c));
	}

protected:
	std::vector<Node*> m_inputs;
	size_t numOutputs, numInputs;
	NodeSet<InputNode> weights;
	InputNode bias;
	std::shared_ptr<VectorMultNode[]> vectorNodes;
};

template<typename ActivationNodeT>
struct Layer : LinearLayer
{
	Layer(const std::vector<Node*>& inputs, size_t nOutputs)
	: LinearLayer(inputs, nOutputs)
	{
		activationNodes = std::shared_ptr<ActivationNodeT[]>(new ActivationNodeT[nOutputs]);

		for(unsigned i = 0; i < nOutputs; ++i)
			activationNodes[i].setParent(vectorNodes.get() + i);
	}

	std::vector<Node*> getOutputNodes()
	{
		std::vector<Node*> result;
		for(size_t i = 0; i < numOutputs; ++i)
			result.push_back(activationNodes.get() + i);
		
		return result;
	}

private:
	std::shared_ptr<ActivationNodeT[]> activationNodes;
};

#endif //LAYERS_H