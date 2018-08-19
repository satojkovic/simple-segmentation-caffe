#!/usr/bin/env sh

echo "Downloading..."

wget -c https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn8s/deploy.prototxt
wget -c http://dl.caffe.berkeleyvision.org/fcn8s-atonce-pascal.caffemodel

echo "Done."