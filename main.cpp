
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

    // Create InputNodes (these get set to values during computation)
    NodeSet<InputNode> inputs(2);
    // Register these InputNodes with the graph
    graph.addInputNodes(inputs.getInputs());

    // Define a fully connected layer (weights) with 2 inputs and 2 outputs
    Layer<SigmoidNode> firstLayer(inputs.getNodes(0, 2), 2);
    firstLayer.randomizeWeights();

    // Get the output nodes of the first layer, feed them into the second layer.
    auto v = firstLayer.getOutputNodes();

    // Let the second layer be the last layer. It has just one output node.
    Layer<SigmoidNode> secondLayer(v, 1);
    secondLayer.randomizeWeights();

    // Register the weight nodes with the graph.
    graph.addParamNodes(firstLayer.getWeightNodes());
    graph.addParamNodes(secondLayer.getWeightNodes());

    // Register the output node with the graph.
    graph.outputNodes = secondLayer.getOutputNodes();

    // optimize params to get an XOR function

    const int TRAINING_SET_SIZE = 4;
    const int N_INPUTS = 2;
    const int N_OUTPUTS = 1;

    double inputValues[N_INPUTS*TRAINING_SET_SIZE] = {
    0,0,
    1,0,
    0,1,
    1,1
    };

    double expectedOutputs[N_OUTPUTS*TRAINING_SET_SIZE] = {
    0,
    1,
    1,
    0
    };

    // Define gradient descent as the optimizer.
    GradientDescent<SquareLoss> optimizer(&graph);

    // Set some hyperparameters.
    optimizer.setLearningRate(1);
    optimizer.setLearningRateDecay(0.9);
    optimizer.setDecayFrequency(5000);

    // Train the network
    optimizer.setTrainingSet(inputValues, expectedOutputs, TRAINING_SET_SIZE);
    optimizer.runEpochs(100000);

    // Test the network
    // A small percent of times (~10%), the test fails.
    // This means the gradient descent gets stuck in a local optimum that is not good.
    // This might be avoidable with better hyperparameters.
    // This is difficult to avoid since we have
    //   - a very small training set (just 4 examples)
    //   - batched gradient descent (makes the gradient smoother)
    //   - unidirectional learning rate decay
    // These factors combined mean that once we are in a valley with a bad local
    // optimum, and our learning rate is too small to jump out of the valley
    // we are completely trapped in the valley.

    for(int j = 0; j < 4; ++j)
    {
        int ind = j*N_INPUTS;
        double output = graph.forwardPass(&inputValues[ind]).at(0);
        std::cout   << "XOR(" << inputValues[ind] << "," << inputValues[ind+1] << ") = "
                    << output << std::endl;
    }

    // Output the weights of the network
    std::cout << "First layer" << std::endl;
    firstLayer.printWeights();
    std::cout << "Second layer" << std::endl;
    secondLayer.printWeights();
    std::cout << std::endl;

    return 0;
}
