#ifndef LAYERS_H
#define LAYERS_H

#include "graph.h"
#include "nodetypes.h"

#include <vector>
#include <memory>
#include <exception>
#include <iostream>

struct InputNodeSet
{
	InputNodeSet(unsigned size)
	: m_size(size)
	{
		if(!size)
			throw new std::exception();

		inputs = std::shared_ptr<InputNode[]>(new InputNode[size]);
	}

	InputNode& at(size_t index)
	{
		if(index > m_size)
			throw new std::exception();

		return inputs[index];
	}

	InputNode* ptrAt(size_t index)
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

	std::vector<InputNode*> getInputs()
	{
		std::vector<InputNode*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = 0; i < m_size; ++i)
			r.push_back(ptr + i);

		return r;
	}

private:
	std::shared_ptr<InputNode[]> inputs;
	unsigned m_size;
};

struct Layer
{
	Layer(std::vector<Node*> inputs, size_t nOutputs)
	: m_inputs(inputs)
	, numInputs(inputs.size())
	, numOutputs(nOutputs)
	, weights(inputs.size()*nOutputs)
	{
		if(!inputs.size())
			throw new std::exception();

		unsigned cols = numInputs;

		nodes = std::shared_ptr<VectorMultNode[]>(new VectorMultNode[nOutputs]);

		for(unsigned r = 0; r < numOutputs; ++r)
		{
			std::vector<Node*> w;
			for(unsigned c = 0; c < cols; ++c)
				w.push_back(weights.ptrAt(r*cols + c));

			nodes[r].setInputs(inputs, w);
		}
	}

	std::vector<InputNode*> getWeightNodes()
	{
		return weights.getInputs();
	}

	std::vector<Node*> getOutputNodes()
	{
		std::vector<Node*> result;
		for(size_t i = 0; i < numOutputs; ++i)
			result.push_back(nodes.get() + i);
		
		return result;
	}

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

private:
	std::vector<Node*> m_inputs;
	size_t numOutputs, numInputs;
	InputNodeSet weights;
	std::shared_ptr<VectorMultNode[]> nodes;
};

#endif //LAYERS_H