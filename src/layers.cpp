
#include "layers.h"
#include <exception>
#include <iostream>

LinearLayer::LinearLayer(const std::vector<Node*>& inputs, size_t nOutputs)
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

std::vector<InputNode*> LinearLayer::getWeightNodes()
{
	return weights.getInputs();
}

std::vector<Node*> LinearLayer::getOutputNodes()
{
	std::vector<Node*> result;
	for(size_t i = 0; i < numOutputs; ++i)
		result.push_back(vectorNodes.get() + i);
	
	return result;
}

InputNode* LinearLayer::getBiasNode() { return &bias; }

void LinearLayer::setWeights(unsigned row, std::vector<double> w)
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

void LinearLayer::randomizeWeights()
{
	for(unsigned i = 0; i < weights.size(); ++i)
	{
		double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		weights.at(i).setInput(r);
	}
}
