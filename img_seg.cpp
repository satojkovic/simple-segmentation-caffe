#include <caffe/caffe.hpp>

using namespace caffe;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt fcn8s-atonce-pascal.caffemodel" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string model_file = argv[1];
    std::string trained_file = argv[2];

    // Load the network
    auto net = new Net<float>(model_file, TEST);
    net->CopyTrainedLayersFrom(trained_file);

    return 0;
}