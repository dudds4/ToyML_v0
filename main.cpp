
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