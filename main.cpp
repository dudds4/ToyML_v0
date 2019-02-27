
#include "loss.h"
#include "batchoptimizer.h"

#include "graph.h"
#include "nodetypes.h"
#include "layers.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

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

	const int TRAINING_SET_SIZE = 4;
	const int N_INPUTS = 2;
	const int N_OUTPUTS = 1;

	float inputValues[N_INPUTS*TRAINING_SET_SIZE] = {
		0,0,
		1,0, 
		0,1, 
		1,1 
	};

	float expectedOutputs[N_OUTPUTS*TRAINING_SET_SIZE] = {
		0,
		1,
		1,
		0
	};

	GradientDescent<SquareLoss> optimizer(&graph);
	optimizer.setTrainingSet(inputValues, expectedOutputs, TRAINING_SET_SIZE);

	// std::cout << "Running epochs..." << std::endl;
	optimizer.runEpochs(10000);

	for(int j = 0; j < 4; ++j)
	{
		int ind = j*N_INPUTS;
		float output = graph.forwardPass(&inputValues[ind]).at(0);
		std::cout 	<< "XOR(" 
					<< inputValues[ind] << "," << inputValues[ind+1]
					<< ") = " << output << std::endl;
	
	}

	return 0;
}