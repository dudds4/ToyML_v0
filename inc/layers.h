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

#endif //LAYERS_H