#include "manualCNN.h"
#include "time.h"

using namespace manualCNN;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

int main()
{
    clock_t t1,t2;
    double duration;
    //t1 = clock();
    //std::cout << "begin_time:" << t1 << std::endl;
    // Set random seed and generate some data
    std::srand(123);
    // Predictors -- each column is an observation
    Matrix x = Matrix::Random(784, 1);
    // Response variables -- each column is an observation
    Matrix y = Matrix::Random(2, 1);

    // Construct a network object
    Network net;

    // Create three layers
    // Layer 1 -- convolutional, input size 20x20x1, 3 output channels, filter size 5x5
    Layer* layer1 = new Convolutional<ReLU>(28, 28, 1, 3, 5, 5);
    // Layer 2 -- max pooling, input size 16x16x3, pooling window size 3x3
    //Layer* layer2 = new MaxPooling<ReLU>(24, 24, 3, 3, 3);
    // Layer 3 -- fully connected, input size 5x5x3, output size 2
    //Layer* layer3 = new FullyConnected<Identity>(8 * 8 * 3, 2);

    // Add layers to the network object
    net.add_layer(layer1);
    //net.add_layer(layer2);
    //net.add_layer(layer3);

    // Set output layer
    net.set_output(new RegressionMSE());

    // Create optimizer object
    RMSProp opt;
    opt.m_lrate = 0.001;

    // (Optional) set callback function object
    VerboseCallback callback;
    net.set_callback(callback);

    // Initialize parameters with N(0, 0.01^2) using random seed 123
    net.init(0, 0.01, 123);

    // Fit the model with a batch size of 100, running 10 epochs with random seed 123
    //net.fit(opt, x, y, 1, 1, 123);

    // Obtain prediction -- each column is an observation

    t1 = clock();
    std::cout << "begin_time:" << t1 << std::endl;

    Matrix pred = net.predict(x);

    //std::cout << pred << std::endl;

    // Layer objects will be freed by the network object,
    // so do not manually delete them
    t2 = clock();
    std::cout << "end_time:" << t2 << std::endl;
    duration = (double)(t2-t1)*1000/CLOCKS_PER_SEC;
    std::cout << "It took me " << duration << " ms." << std::endl;

    return 0;
}
