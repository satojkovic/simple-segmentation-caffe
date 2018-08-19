# simple-segmentation-caffe

## How to Use

1. ```$ sh get_prototxt_and_caffemodel.sh```
1. compile img_seg.cpp
    * For example on Mac OSX  
    ```$ clang++ pred.cpp -std=c++11 -lboost_system -lcaffe -lglog -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o pred``` 
1. Usage  
    ```./img_seg deploy.prototxt fcn8s-atonce-pascal.caffemodel image.jpg```