To run ILSVRC experiments:

1. Download ILSVRC dataset into "ILSVRC_data" folder and preprocess it so that all images have 
   shape (227, 227, 3).

   The dataset I used is ILSVRC 2012 Validation Set:
   http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

   Command for cropping images:
   mogrify *.JPEG -resize "227x227^" -gravity center -crop 227x227+0+0 +repage
   

2. Download the caffe model file for Alexnet, VGG16 or GoogLenet into the corresponding subdirectory under 
   "models" folder.

   Alexnet model file:
   http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

   VGG16 model file:
   http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

   GoogLenet model file:
   http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

   To use other models, simply add a branch in "caffe_models.py"