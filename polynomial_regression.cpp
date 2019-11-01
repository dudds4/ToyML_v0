
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

template<typename T>
void tprint(T item)
{
    std::cout << item ;
}

template<typename T, typename ... Types>
void tprint(T item, Types ... args)
{
    std::cout << item << ' ';
    tprint(args...);
}

double randInRange(double lower, double upper)
{
    return rand() * ((upper - lower) / RAND_MAX) + lower;
}

void generateCoefficients(double* coefficients, int length)
{
    for(int i = 0; i < length; ++i)
        coefficients[i] = randInRange(-10, 10);
}

void samplePoints(double points[][2], int N_POINTS, double* coefficients, int length)
{
    for(int i = 0; i < N_POINTS; ++i)
    {
        double x = randInRange(-100, 100);
        double product = 1;
        double y = 0;

        for(int j = 0; j < length; j++)
        {
            y += coefficients[j]*product;
            product *= x;            
        }

        points[i][0] = x;
        points[i][1] = y;
    }    
}

int main()
{
    srand(time(NULL));

    int n_coeffs = 4;
    int N_POINTS = 500;

    tprint("Generating 3rd degree random polynomial...\n");

    double coefficients[n_coeffs];
    generateCoefficients(coefficients, n_coeffs);

    tprint("Sampling from the polynomial...\n");

    double points[N_POINTS][2];
    samplePoints(points, N_POINTS, coefficients, n_coeffs);

    tprint("Generating the graph...\n");
    Graph graph;

    int inputs_per_point = n_coeffs-1;
    NodeSet<InputNode> inputs(inputs_per_point);
    LinearLayer layer(inputs.getNodes(0, inputs_per_point), 1);
    layer.randomizeWeights();

    graph.addInputNodes(inputs.getInputs());
    graph.addParamNodes(layer.getWeightNodes());
    graph.outputNodes = layer.getOutputNodes();

    tprint("Processing the inputs to mesh with graph format...\n");
    
    double inputValues[inputs_per_point*N_POINTS];
    double expectedOutputs[N_POINTS];

    for(int i = 0; i < N_POINTS; ++i)
    {
        double x = points[i][0];
        double product = x;

        for(int j = 0; j < inputs_per_point; ++j)
        {
            inputValues[i*inputs_per_point + j] = product;
            product *= x;
        }

        expectedOutputs[i] = points[i][1];
    }

    tprint("Fitting...\n");

    GradientDescent<SquareLoss> optimizer(&graph);
    optimizer.setTrainingSet(inputValues, expectedOutputs, N_POINTS);
    optimizer.setGradientClipping(500);

    for(int i = 0; i < 5; i++)
    {
        optimizer.setLearningRate(0.0001);
        int n = 2000;
        optimizer.runEpochs(n);
        std::cout << "\t" <<  n * (i+1) << " epochs...\n";    
    }

    std::cout << optimizer.getLearningRate() << std::endl;
    auto weights = layer.getWeightNodes();

    std::cout << "fit coefficients: " << std::endl;
    int w_idx=0;
    auto weight = weights.back();
    std::cout << "a" << w_idx++ << "=" << weight->getInput() << std::endl;
    for(int i = 0; i < weights.size()-1; ++i)
        std::cout << "a" << w_idx++ << "=" << weights[i]->getInput() << std::endl;

    std::cout << "actual coefficients: " << std::endl;
    w_idx=0;
    for(auto coeff : coefficients)
        std::cout << "a" << w_idx++ << "=" << coeff << std::endl;
    
    return 0;
}