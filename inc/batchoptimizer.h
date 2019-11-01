#ifndef BATCH_OPTIMIZER_H
#define BATCH_OPTIMIZER_H

#include "graph.h"
#include "nodetypes.h"
#include <cstring>
#include <iostream>

typedef std::vector<double> floatset;

template<template<typename T> class OptimT, typename LossT>
struct BatchOptimizer
{
	typedef OptimT<LossT> OptimizerT;

	void setGraph(Graph *g) { 
		graph = g;
		nParams = graph->paramNodes.size();
		paramDerivs.resize(nParams);
	}

	void setTrainingSet(double* in, double* out, unsigned n)
	{ 
		inputs = in;
		outputs = out;
		setSize = n;
	}

	void runEpochs(unsigned iterations) { for(int i = 0; i < iterations; ++i) runEpoch(); }

	void updateParamsInterface() { static_cast<OptimizerT*>(this)->updateParams(); }

	void runEpoch()
	{
		unsigned inW = graph->inputNodes.size();
		unsigned outW = graph->outputNodes.size();

		memset(paramDerivs.data(), 0, sizeof(double)*nParams);

		double overallError = 0;

		double *inPtr = inputs;
		double *outPtr = outputs;

		// compute summed derivative
		for(unsigned j = 0; j < setSize; ++j)
		{
			auto outputs = graph->forwardPass(inPtr);
			overallError += LossT::loss(outputs.data(), outPtr, outW);
			auto baseDeriv = LossT::derivative(outputs.data(), outPtr, outW);
			graph->backProp(baseDeriv);

			for(unsigned k = 0; k < nParams; ++k)
				paramDerivs[k] += graph->paramNodes[k]->getDerivative(0) / (double)setSize;

			inPtr += inW;
			outPtr += outW;

		}

		if(overallError > lastOverallError)
			learningRate /= 2;

		lastOverallError = overallError;

		// gradient clipping
		if(maxGradient > 0)
		{
			for(unsigned k = 0; k < nParams; ++k)
			{	
				if(paramDerivs[k] > maxGradient) paramDerivs[k] = maxGradient;
				else if(paramDerivs[k] < -maxGradient) paramDerivs[k] = -maxGradient;
			}
		}

		// update params
		updateParamsInterface();
	}

	void setGradientClipping(double maxGrad=-1) { maxGradient = maxGrad; }
	void setLearningRate(double r) { learningRate = r; } 
	double getLearningRate() { return learningRate; } 

protected:
	Graph *graph;
	double* inputs; 
	double* outputs;
	unsigned setSize;

	double lastOverallError = 0;
	double learningRate = 0.2;
	double maxGradient = -1;

	std::vector<double> paramDerivs;
	unsigned nParams;
};


template<typename LossT>
struct GradientDescent : public BatchOptimizer<GradientDescent, LossT>
{
	GradientDescent(Graph *g) {
		this->setGraph(g);
	}

	void updateParams()
	{
		unsigned nParams = this->graph->paramNodes.size();
		for(unsigned k = 0; k < nParams; ++k)
		{
			auto pNode = this->graph->paramNodes[k];
			
			double w = pNode->getInput();
			pNode->setInput(w - this->learningRate*this->paramDerivs[k]);
		}		
	}
};

#endif