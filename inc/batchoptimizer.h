#ifndef BATCH_OPTIMIZER_H
#define BATCH_OPTIMIZER_H

#include "graph.h"
#include "nodetypes.h"
#include <cstring>

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

#endif