import os
import numpy as np
from PIL import Image
from utils import *

class PredDiffAnalyzer:
    def __init__(self, classifier, sampler, batch_size):
        self.classifier = classifier
        self.sampler = sampler
        self.batch_size = batch_size

    def get_rel_vects(self, X):
        win_size = self.sampler.win_size
        height, width, num_channels = X[0].shape
        num_feats = height * width
        counts = np.zeros((X.shape[0], num_feats), dtype=np.int)
        rel_vectors = np.zeros((X.shape[0], num_feats))
        feats_ind = np.reshape(np.arange(num_feats*num_channels), X[0].shape)
        classifier_output = self.classifier.classify(X)
        true_probs = np.max(classifier_output, axis=1)
        label_ind = np.argmax(classifier_output, axis=1)

        for k in range(X.shape[0]):
            print('Computing relevence vector for image {}...'.format(k+1))
            cond_probs = []
            new_Xs = np.empty((self.batch_size, height, width, num_channels), dtype=np.float32)
            idx = 0
            for i in range(height - win_size + 1):
                for j in range(width - win_size + 1):
                    new_X = np.copy(X[k])
                    window_ind = feats_ind[i:i+win_size, j:j+win_size, :].ravel()
                    cond_mean = self.sampler.get_cond_mean(window_ind, X[k])
                    print('\tConditional mean taken for image {} at {},{}...'.format(k+1, i, j))
                    new_X.put(window_ind, cond_mean.ravel())
                    if idx < self.batch_size:
                        new_Xs[idx] = new_X
                        idx += 1
                    else:
                        cond_probs += list(self.classifier.classify(new_Xs)[:, label_ind[k]])
                        new_Xs[0] = new_X
                        idx = 1
            cond_probs += list(self.classifier.classify(new_Xs)[:idx, label_ind[k]])
            for i in range(height - win_size + 1):
                for j in range(width - win_size + 1):
                    idx = i * (height - win_size + 1) + j
                    feats_ind_2d = np.reshape(np.arange(num_feats), (height, width))
                    window_ind_2d = feats_ind_2d[i:i+win_size, j:j+win_size].ravel()
                    rel_vectors[k][window_ind_2d] += self.sampler.log_odds(
                        true_probs[k]) - self.sampler.log_odds(cond_probs[idx])
                    counts[k][window_ind_2d] += 1
        return rel_vectors / counts

    def visualize(self, input_path, output_path):
        valid_suffices = ['png', 'jpg', 'jpeg']
        img_list = os.listdir(input_path)

        for img_file in img_list:
            if not any([img_file.lower().endswith(s) for s in valid_suffices]):
                img_list.remove(img_file)

        height, width, num_channels = self.classifier.input_size
        X = np.empty((0, height, width, num_channels))
        X_orig = np.empty((0, height, width, num_channels))
        X_filenames = []
        for i in img_list:
            img = np.float32(Image.open(os.path.join(input_path, i)))
            if img.shape[0] >= height and img.shape[1] >= width:
                cropped_img = center_crop(img, height, width)
                X_orig = np.vstack((X_orig, np.array([cropped_img])))
                preprocessed_img = np.rollaxis(self.classifier.preprocess(cropped_img), 0, 3)
                X = np.vstack((X, np.array([preprocessed_img])))
                X_filenames.append(i)
            else:
                print('Skip {}'.format(i))
        rel_vects = self.get_rel_vects(X).reshape((-1, height, width))
        for i in range(rel_vects.shape[0]):
            print('Visualizing image {}...'.format(i+1))
            filename, suffix = X_filenames[i].split('.')
            plot_heatmap(rel_vects[i], output_path, filename + '_fhm' + '.' + suffix)
            plot_overlayed_heatmap(X_orig[i].reshape(
                (height, width, num_channels)), rel_vects[i], output_path, filename + '_foh' + '.' + suffix)