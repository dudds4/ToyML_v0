#include "graph.h"
#include "nodetypes.h"
#include "layers.h"

#include <iostream>
#include <cmath>
#include <cstring>

#include <cstdlib>
#include <ctime>

struct SquareLoss
{
	static float loss(float yout, float yexpected)
	{
		float x = yout - yexpected;
		return x * x;
	}

	static float derivative(float yout, float yexpected)
	{
		return 2 * (yout - yexpected);
	}
};

typedef std::vector<float> floatset;

template<template<typename T> class OptimT, typename LossT>
struct BatchOptimizer
{
	typedef OptimT<LossT> OptimizerT;

	void setGraph(Graph *g) { 
		graph = g;
		nParams = graph->paramNodes.size();
		paramDerivs.resize(nParams);
	}

	void setTrainingSet(std::vector<float> *in, float* out, unsigned n)
	{ 
		inputs = in;
		outputs = out;
		setSize = n;
	}

	void runEpochs(unsigned iterations) { for(int i = 0; i < iterations; ++i) runEpoch(); }

	void updateParamsInterface() { static_cast<OptimizerT*>(this)->updateParams(); }

	void runEpoch()
	{
		memset(paramDerivs.data(), 0, sizeof(float)*nParams);
		float overallError = 0;

		// compute summed derivative
		for(unsigned j = 0; j < setSize; ++j)
		{
			float output = graph->forwardPass(inputs[j]).at(0);

			overallError += LossT::loss(output, outputs[j]);
			float baseDeriv = LossT::derivative(output, outputs[j]);

			graph->backProp(0, baseDeriv);

			for(unsigned k = 0; k < nParams; ++k)
				paramDerivs[k] += graph->paramNodes[k]->getDerivative(0);

		}

		if(overallError > lastOverallError)
			learningRate /= 2;

		lastOverallError = overallError;

		// update params
		updateParamsInterface();
	}

	void setLearningRate(float r) { learningRate = r; } 

protected:

	Graph *graph;
	std::vector<float> *inputs; 
	float* outputs;
	unsigned setSize;

	float lastOverallError = 0;
	float learningRate = 0.2;

	std::vector<float> paramDerivs;
	unsigned nParams;
};


template<typename LossT>
struct GradientDescent : public BatchOptimizer<GradientDescent, LossT>
{
	GradientDescent(Graph *g) {
		this->graph = g;
	}

	void updateParams()
	{
		unsigned nParams = this->graph->paramNodes.size();

		for(unsigned k = 0; k < nParams; ++k)
		{
			auto pNode = this->graph->paramNodes[k];
			float w = pNode->getInput();
			pNode->setInput(w - this->learningRate*this->paramDerivs[k]);
		}		
	}
};

int main()
{
	srand(time(NULL));

	// specify the graph
	Graph graph;

	NodeSet<InputNode> inputs(2);
	graph.addInputNodes(inputs.getInputs());

	Layer<SigmoidNode> firstLayer(inputs.getNodes(0, 2), 2);
	firstLayer.randomizeWeights();

	auto v = firstLayer.getOutputNodes();	

	Layer<SigmoidNode> secondLayer(v, 1);
	secondLayer.randomizeWeights();

	graph.addParamNodes(firstLayer.getWeightNodes());
	graph.addParamNodes(secondLayer.getWeightNodes());

	graph.outputNodes = secondLayer.getOutputNodes();

	// optimize params to get an XOR function

	floatset inputValues[4];
	inputValues[0] = {0,0,1,1};
	inputValues[1] = {1,0,1,1};
	inputValues[2] = {0,1,1,1};
	inputValues[3] = {1,1,1,1};

	floatset expectedOutputs = {0,1,1,0};

	GradientDescent<SquareLoss> optimizer(&graph);
	optimizer.setTrainingSet(inputValues, expectedOutputs.data(), expectedOutputs.size());
	optimizer.runEpochs(10000);

	for(int j = 0; j < 4; ++j)
	{
		float output = graph.forwardPass(inputValues[j]).at(0);
		std::cout 	<< "XOR(" 
					<< inputValues[j][0] << "," << inputValues[j][1]
					<< ") = " << output << std::endl;
	
	}

	return 0;
}