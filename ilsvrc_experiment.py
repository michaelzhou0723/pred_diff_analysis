import os
import numpy as np
from PIL import Image
from pred_diff_analysis import PredDiffAnalyzer
from sampler import CondGaussSampler3d
from caffe_models import Classifier
from utils import center_crop

class Imagenet_Sampler(CondGaussSampler3d):
    def __init__(self, classifier, win_size, padding, num_samples_fit=20000):
        self.classifier = classifier
        self.image_size = classifier.input_size
        super().__init__(win_size, padding, self.image_size, classifier.name, num_samples_fit)

    def load_data(self):
        img_path = './ILSVRC_data'
        filenames = os.listdir(img_path)
        X = np.empty((self.num_samples_fit, *self.image_size))
        for i in range(self.num_samples_fit):
            img = np.float32(Image.open(os.path.join(img_path, filenames[i])))
            if img.shape[0] >= self.image_size[0] and img.shape[1] >= self.image_size[1]:
                cropped_img = center_crop(img, self.image_size[0], self.image_size[1])
                X[i] = np.rollaxis(self.classifier.preprocess(cropped_img), 0, 3)
            if (i+1) % 1000 == 0:
                print('{} images are loaded'.format(i+1))
        return X

    def log_odds(self, prob):
        ilsvrc_train = 1300000
        ilsvrc_labels = 1000
        laplace_prob = (prob * ilsvrc_train + 1) / (ilsvrc_train + ilsvrc_labels)
        return np.log2(laplace_prob / (1 - laplace_prob))

    def get_cond_mean(self, win_ind, feat_vector_orig):
        feat_vector = np.copy(feat_vector_orig)
        patches, inpatch_ind = self._get_surr_patch(feat_vector, win_ind)

        patches = np.array([patches[:, :, i].ravel() for i in range(3)])
        inpatch_ind = inpatch_ind.ravel()
        cond_means, _ = self._get_cond_params(patches, inpatch_ind)
        
        return np.array(cond_means).swapaxes(0, 1)


if __name__ == '__main__':
    # valid names are alexnet, vgg16 and googlenet
    imagenet_classifier = Classifier('vgg16')
    imagenet_sampler = Imagenet_Sampler(imagenet_classifier, win_size=10, padding=4)
    # Maximum batch size for Alexnet on K80 is 2000, for VGG16 is 160
    analyzer = PredDiffAnalyzer(imagenet_classifier, imagenet_sampler, batch_size=160)
    analyzer.visualize('./inputs', './outputs')