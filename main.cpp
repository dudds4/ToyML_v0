#include "graph.h"
#include "nodetypes.h"
#include "layers.h"

#include <iostream>
#include <cmath>

int main()
{
	// specify the graph

	Graph graph;
	
	// inputs x1, x2
	// biases b1, b2

	InputNodeSet inputs(4);
	graph.addInputNodes(inputs.getInputs());

	Layer firstLayer(inputs.getNodes(0, 3), 2);

	firstLayer.setWeights(0, {1,1,1});
	firstLayer.setWeights(1, {1,1,1});

	auto v = firstLayer.getOutputNodes();
	v.push_back(inputs.getNodes(3,4).at(0));

	Layer secondLayer(v, 1);
	secondLayer.setWeights(0, {1,1,1});

	graph.addParamNodes(firstLayer.getWeightNodes());
	graph.addParamNodes(secondLayer.getWeightNodes());

	graph.outputNodes = secondLayer.getOutputNodes();

	// optimize params to get an XOR function

	float goalOutput = 0.6;
	float learningRate = 0.01;

	std::vector<float> inputValues[4];
	inputValues[0] = {0,0,1,1};
	inputValues[1] = {1,0,1,1};
	inputValues[2] = {0,1,1,1};
	inputValues[3] = {1,1,1,1};

	float expectedOutputs[4] = {0,1,1,0};
	float lastOverallError = 9999999999;

	// let's try and train this graph
	for(int i = 0; i < 1000; ++i)
	{
		float overallError = 0;
		bool shouldPrint = i % 100 == 0;

		for(int j = 0; j < 4; ++j)
		{
			graph.setInputs(inputValues[j]);

			graph.traverse();
			float output = graph.getOutput(0);

			if(shouldPrint)
				std::cout 	<< "XOR(" 
							<< inputValues[j][0] << "," << inputValues[j][1]
							<< ") = " << output << std::endl;

			overallError += pow(output-expectedOutputs[j], 2);
			
			graph.backProp(0);

			auto paramUpdater = [=](float w, float deriv)
			{
				float sign = output > expectedOutputs[j] ? 1 : -1;
				return w - sign * deriv * learningRate;			
			};

			graph.updateParams(paramUpdater);			
		}

		if(overallError > lastOverallError)
			learningRate /= 2;

		lastOverallError = overallError;

		// std::cout << "Error: " << overallError << std::endl;
	}

	return 0;
}