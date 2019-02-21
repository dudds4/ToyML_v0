#include "graph.h"
#include "nodetypes.h"
#include "layers.h"

#include <iostream>

int main()
{
	// specify the graph

	Graph graph;
	
	InputNodeSet inputs(2);

	Layer layer(inputs.getNodes(), 1);
	layer.setWeights(0, {1,1});
	graph.addInputNodes(inputs.getInputs());

	graph.addParamNodes(layer.getWeightNodes());
	graph.outputNodes = layer.getOutputNodes();

	graph.setInputs({4,1});

	// optimize params on something

	float goalOutput = 0.6;
	float learningRate = 0.01;
	float lastError = 9999999999;

	// let's try and train this graph
	for(int i = 0; i < 100; ++i)
	{
		graph.traverse();
		float output = graph.getOutput(0);
		
		{
			float error = goalOutput - output;
			error = error * error;
			
			if(error > lastError)
				learningRate /= 2;

			lastError = error;			
		}

		std::cout << "Computation result: " << output << std::endl;
		
		graph.backProp(0);

		auto paramUpdater = [=](float w, float deriv)
		{
			float sign = output > goalOutput ? 1 : -1;
			return w - sign * deriv * learningRate;			
		};

		graph.updateParams(paramUpdater);

	}

	return 0;
}