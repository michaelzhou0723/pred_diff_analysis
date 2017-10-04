import os
import caffe
import numpy as np

class Classifier:
    def __init__(self, netname, gpu=True):
        if gpu:
            caffe.set_mode_gpu()
        root_path = './models'
        if netname == 'vgg16':
            model_path = os.path.join(root_path, 'vgg16')
            proto_path = os.path.join(model_path, 'VGG_ILSVRC_16_layers_deploy.prototxt')
            param_path = os.path.join(model_path, 'VGG_ILSVRC_16_layers.caffemodel')
            mean = np.float32([103.939, 116.779, 123.68])
        elif netname == 'googlenet':
            model_path = os.path.join(root_path, 'googlenet')
            proto_path = os.path.join(model_path, 'deploy.prototxt')
            param_path = os.path.join(model_path, 'bvlc_googlenet.caffemodel')
            mean = np.float32([104.0, 117.0, 123.0])
        elif netname == 'alexnet':
            model_path = os.path.join(root_path, 'alexnet')
            proto_path = os.path.join(model_path, 'deploy.prototxt')
            param_path = os.path.join(model_path, 'bvlc_reference_caffenet.caffemodel')
            mean = np.float32([104.0, 117.0, 123.0])
        self.net = caffe.Classifier(proto_path, param_path, caffe.TEST, channel_swap = (2,1,0), mean = mean)       
        self.input_size = (*self.net.crop_dims, 3)
        self.name = netname   

    def classify(self, X, blobname='prob'):
        if np.ndim(X) == 4 and X.shape[-1] == 3:
            X = np.rollaxis(X, 3, 1)
        if self.net.blobs['data'].data.shape[0] != X.shape[0]:
            self.net.blobs['data'].reshape(*(X.shape))
        self.net.forward_all(data=X)
        return np.copy(self.net.blobs[blobname].data[:])

    def preprocess(self, img):
        return self.net.transformer.preprocess('data', img)
