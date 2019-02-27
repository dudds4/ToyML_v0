#ifndef LAYERS_H
#define LAYERS_H

#include "graph.h"
#include "nodeset.h"
#include "nodetypes.h"

#include <memory>
#include <vector>

struct LinearLayer
{
	LinearLayer(const std::vector<Node*>& inputs, size_t nOutputs);

	std::vector<InputNode*> getWeightNodes();
	std::vector<Node*> getOutputNodes();
	InputNode* getBiasNode();
	void setWeights(unsigned row, std::vector<float> w);
	void randomizeWeights();

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


struct SoftMaxLayer
{
	SoftMaxLayer() = delete;
	SoftMaxLayer(const std::vector<Node*>& inputs)
	: maxNode(inputs)
	, inverseNode(&maxNode)
	{		
		unsigned n = inputs.size();
		multiplicationNodes = std::shared_ptr<MultiplicationNode[]>(new MultiplicationNode[n]);

		for(unsigned i = 0; i < n; ++i)
		{
			multiplicationNodes[i].setParents({inputs.at(i), &inverseNode});
		}
		numOutputs = n;
	}

	std::vector<Node*> getOutputNodes()
	{
		std::vector<Node*> result;
		for(size_t i = 0; i < numOutputs; ++i)
			result.push_back(multiplicationNodes.get() + i);
		
		return result;
	}

private:
	MaxNode maxNode;
	InverseNode inverseNode;
	std::shared_ptr<MultiplicationNode[]> multiplicationNodes;
	unsigned numOutputs;
};

#endif //LAYERS_H