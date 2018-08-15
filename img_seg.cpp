#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

void Preprocess(const cv::Mat &img, int num_channels, cv::Size input_geometry,
                std::vector<cv::Mat> *input_channels)
{
    cv::Mat sample;
    if (img.channels() == 3 && num_channels == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && num_channels == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && num_channels == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels == 3) {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
        sample = img;
    }

    cv::Mat sample_resized;
    if (sample.size() != input_geometry) {
        cv::resize(sample, sample_resized, input_geometry);
    } else {
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if (num_channels == 3) {
        sample_resized.convertTo(sample_float, CV_32FC3);
    } else {
        sample_resized.convertTo(sample_float, CV_32FC1);
    }

    cv::split(sample_float, *input_channels);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt fcn8s-atonce-pascal.caffemodel"
                  << " image.jpg" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string model_file = argv[1];
    std::string trained_file = argv[2];
    std::string img_file = argv[3];

    // Load the network
    auto net = new Net<float>(model_file, TEST);
    net->CopyTrainedLayersFrom(trained_file);
    Blob<float> *input_layer = net->input_blobs()[0];
    int num_channels_ = input_layer->channels();
    cv::Size input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    // Load the image
    cv::Mat img = cv::imread(img_file);
    CHECK(!img.empty()) << "Unable to decode image. " << img_file;

    // Shape for input (data blob is N x C x H x W)
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net->Reshape();

    // Set input data
    std::vector<cv::Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for(int i = 0; i < input_layer->channels(); i++) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += (width * height);
    }
    Preprocess(img, num_channels_, input_geometry_, &input_channels);

    // Forward prop
    net->Forward();

    // Get outputs
    Blob<float> *output_layer = net->output_blobs()[0];
    std::cout << "output_blob(n, c, h, w) = " << output_layer->num() << ", " << output_layer->channels()
        << ", " << output_layer->height() << ", " << output_layer->width() << std::endl;

    cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), 
                                            CV_32F, const_cast<float *>(output_layer->mutable_cpu_data()));
    merged_output_image.convertTo(merged_output_image, CV_8U);
    cv::imshow("TEST", merged_output_image);
    cv::waitKey(0);

    return 0;
}