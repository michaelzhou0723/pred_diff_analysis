import os
import random
import numpy as np

class CondGaussSampler2d:
    def __init__(self, win_size, padding, image_size, sampler_name, num_samples_fit):
        self.win_size = win_size
        self.padding = padding
        self.image_size = image_size
        self.num_samples_fit = num_samples_fit
        self.patch_size = win_size + 2 * padding
        # path for pre-computed parameters
        self.params_path = os.path.join('./params/', sampler_name)

        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
        self.mean_vect, self.cov_mat = self._get_joint_params()
        self.minmax_vals = self._get_minmax_vals()

    def load_data(self):
        raise NotImplementedError

    def log_odds(self):
        raise NotImplementedError

    def _get_joint_params(self):
        mean_path = os.path.join(self.params_path, 'mean_{}_joint.npy'.format(self.patch_size))
        cov_path = os.path.join(self.params_path, 'cov_{}_joint.npy'.format(self.patch_size))

        if os.path.exists(mean_path) and os.path.exists(cov_path):
            mean = np.load(mean_path)
            cov = np.load(cov_path)
        else:
            X = self.load_data()
            patches = np.empty((0, self.patch_size**2))
            for _ in range(self.num_samples_fit // X.shape[0]):
                idx = random.sample(range((self.image_size[0]-self.patch_size+1)*(self.image_size[1]-self.patch_size+1)), 1)[0]
                idx = np.unravel_index(idx, (self.image_size[0]-self.patch_size+1, self.image_size[1]-self.patch_size+1))
                patch = X[:, idx[0]:idx[0]+self.patch_size, idx[1]:idx[1]+self.patch_size]
                patches = np.vstack((patches, patch.reshape((-1, self.patch_size**2))))
            if self.num_samples_fit % X.shape[0]:
                idx = random.sample(range((self.image_size[0]-self.patch_size+1)*(self.image_size[1]-self.patch_size+1)), 1)[0]
                idx = np.unravel_index(idx, (self.image_size[0]-self.patch_size+1, self.image_size[1]-self.patch_size+1))
                patch = X[:self.num_samples_fit, idx[0]:idx[0]+self.patch_size, idx[1]:idx[1]+self.patch_size]
                patches = np.vstack((patches, patch.reshape((-1, self.patch_size**2))))
            mean = np.mean(patches, axis=0)
            cov = np.cov(patches.T)
        np.save(mean_path, mean)
        np.save(cov_path, cov)
        return mean, cov

    def _get_minmax_vals(self):
        minmax_path = os.path.join(self.params_path, 'minmax.npy')

        if os.path.exists(minmax_path):
            minmax_vals = np.load(minmax_path)
        else:
            X = self.load_data()
            minmax_vals = np.array([np.min(X, axis=0), np.max(X, axis=0)])
            np.save(minmax_path, minmax_vals)
        return minmax_vals

    def _get_cond_params(self, patch, inpatch_ind):
        mean_path = os.path.join(self.params_path, 'mean_{}_{}_{}_cond.npy'.format(
            self.patch_size, self.win_size, inpatch_ind[0]))
        cov_path = os.path.join(self.params_path, 'cov_{}_{}_{}_cond.npy'.format(
            self.patch_size, self.win_size, inpatch_ind[0]))

        if os.path.exists(mean_path) and os.path.exists(cov_path):
            cond_mean = np.load(mean_path)
            cond_cov = np.load(cov_path)
        else:
            x2 = np.delete(patch, inpatch_ind)
            mu1 = np.take(self.mean_vect, inpatch_ind)
            mu2 = np.delete(self.mean_vect, inpatch_ind)
            cov11 = self.cov_mat[inpatch_ind][:, inpatch_ind]
            cov12 = np.delete(self.cov_mat[inpatch_ind, :], inpatch_ind, axis=1)
            cov21 = np.delete(self.cov_mat[:, inpatch_ind], inpatch_ind, axis=0)
            cov22 = np.delete(np.delete(self.cov_mat, inpatch_ind, axis=0), inpatch_ind, axis=1)
            dotprod = np.dot(cov12, np.linalg.pinv(cov22))
            cond_mean = mu1 + dotprod.dot(x2-mu2)
            cond_cov = cov11 - dotprod.dot(cov21)
            np.save(mean_path, cond_mean)
            np.save(cov_path, cond_cov)
        return cond_mean, cond_cov

    def get_samples(self, win_ind, feat_vector_orig, num_samples):
        feat_vector = np.copy(feat_vector_orig).reshape((self.image_size[0], self.image_size[1]))
        patch, inpatch_ind = self._get_surr_patch(feat_vector, win_ind)

        patch = patch.ravel()
        inpatch_ind = inpatch_ind.ravel()
        cond_mean, cond_cov = self._get_cond_params(patch, inpatch_ind)

        samples = np.random.multivariate_normal(cond_mean, cond_cov, num_samples)

        samples = samples.reshape((num_samples, -1))
        minvals = self.minmax_vals[0].ravel()[win_ind]
        maxvals = self.minmax_vals[1].ravel()[win_ind]
        for i in range(num_samples):
            samples[i][samples[i]<minvals] = minvals[samples[i]<minvals]
            samples[i][samples[i]>maxvals] = maxvals[samples[i]>maxvals]
        return samples

    def _get_surr_patch(self, feat_vector, win_ind):
        height = self.image_size[0]
        width = self.image_size[1]
        patch_ind = np.array(range(self.patch_size**2)).reshape((self.patch_size, self.patch_size))
        win_ind = np.array(np.unravel_index(win_ind, self.image_size)).T

        top_padding = win_ind[0][0]
        left_padding = win_ind[0][1]
        bottom_padding = height - win_ind[-1][0] - 1
        right_padding = width - win_ind[-1][1] - 1

        rel_height = top_padding - height + self.patch_size
        rel_width = left_padding - width + self.patch_size

        if top_padding <= self.padding:
            if left_padding <= self.padding:
                patch = feat_vector[:self.patch_size][:, :self.patch_size]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[:self.patch_size][:, width - self.patch_size:]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[:self.patch_size][:, left_padding -
                                                      self.padding:left_padding - 
                                                      self.padding + self.patch_size]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, self.padding:self.padding + self.win_size]
        elif bottom_padding <= self.padding:
            if left_padding <= self.padding:
                patch = feat_vector[height-self.patch_size:][:, :self.patch_size]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[height-self.patch_size:][:, width - self.patch_size:]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[height-self.patch_size:][:, left_padding -
                                                             self.padding:left_padding -
                                                             self.padding + self.patch_size]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, self.padding:self.padding + self.win_size]
        else:
            patch_height = top_padding - self.padding
            patch_width = left_padding - self.padding
            if left_padding <= self.padding:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, :self.patch_size]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, width - self.patch_size:]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, patch_width:patch_width + self.patch_size]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, self.padding:self.padding + self.win_size]

        return patch, inpatch_ind


class CondGaussSampler3d(CondGaussSampler2d):
    def _get_joint_params(self):
        mean_path = os.path.join(self.params_path, 'mean_{}_joint.npy'.format(self.patch_size))
        cov_path = os.path.join(self.params_path, 'cov_{}_joint.npy'.format(self.patch_size))

        if os.path.exists(mean_path) and os.path.exists(cov_path):
            mean = np.load(mean_path)
            cov = np.load(cov_path)
        else:
            X = self.load_data()
            patches = np.empty((0, self.patch_size**2, 3))
            for _ in range(self.num_samples_fit // X.shape[0]):
                idx = random.sample(range((self.image_size[0]-self.patch_size+1)*(self.image_size[1]-self.patch_size+1)), 1)[0]
                idx = np.unravel_index(idx, (self.image_size[0]-self.patch_size+1, self.image_size[1]-self.patch_size+1))
                patch = X[:, idx[0]:idx[0]+self.patch_size, idx[1]:idx[1]+self.patch_size, :]
                patches = np.vstack((patches, patch.reshape((-1, self.patch_size**2, 3))))
            if self.num_samples_fit % X.shape[0]:
                idx = random.sample(range((self.image_size[0]-self.patch_size+1)*(self.image_size[1]-self.patch_size+1)), 1)[0]
                idx = np.unravel_index(idx, (self.image_size[0]-self.patch_size+1, self.image_size[1]-self.patch_size+1))
                patch = X[:self.num_samples_fit, idx[0]:idx[0]+self.patch_size, idx[1]:idx[1]+self.patch_size, :]
                patches = np.vstack((patches, patch.reshape((-1, self.patch_size**2, 3))))
            mean = np.mean(patches, axis=0)
            cov = np.array([np.cov(patches[:, :, i].T) for i in range(3)])
        np.save(mean_path, mean)
        np.save(cov_path, cov)
        print('Finish computing joint parameters')
        return mean, cov

    def _get_cond_params(self, patches, inpatch_ind):
        dotprod_path = os.path.join(self.params_path, 'dotprod_{}_{}_{}_cond.npy'.format(
            self.patch_size, self.win_size, inpatch_ind[0]))
        condcov_path = os.path.join(self.params_path, 'condconv_{}_{}_{}_cond.npy'.format(
            self.patch_size, self.win_size, inpatch_ind[0]))
        
        condmeans, condcovs, dotprods = [], [], []

        if os.path.exists(condcov_path) and os.path.exists(dotprod_path):
            condcovs = np.load(condcov_path)
            dotprods = np.load(dotprod_path)
            for i in range(3):
                patch = patches[i]
                x2 = np.delete(patch, inpatch_ind)
                mu1 = np.take(self.mean_vect[:, i], inpatch_ind)
                mu2 = np.delete(self.mean_vect[:, i], inpatch_ind)
                condmeans.append(mu1 + dotprods[i].dot(x2-mu2))  
        else:
            for i in range(3):
                patch = patches[i]
                x2 = np.delete(patch, inpatch_ind)
                mu1 = np.take(self.mean_vect[:, i], inpatch_ind)
                mu2 = np.delete(self.mean_vect[:, i], inpatch_ind)
                               
                cov11 = self.cov_mat[i][inpatch_ind][:, inpatch_ind]
                cov12 = np.delete(self.cov_mat[i][inpatch_ind, :], inpatch_ind, axis=1)
                cov21 = np.delete(self.cov_mat[i][:, inpatch_ind], inpatch_ind, axis=0)
                cov22 = np.delete(np.delete(self.cov_mat[i], inpatch_ind, axis=0), inpatch_ind, axis=1)
                dotprod = np.dot(cov12, np.linalg.inv(cov22))
                condcovs.append(cov11 - dotprod.dot(cov21))
                dotprods.append(dotprod)
                condmeans.append(mu1 + dotprods[i].dot(x2-mu2))
            np.save(dotprod_path, np.array(dotprods))
            np.save(condcov_path, np.array(condcovs))          

        return condmeans, condcovs

    def get_samples(self, win_ind, feat_vector, num_samples):
        patches, inpatch_ind = self._get_surr_patch(feat_vector, win_ind)

        patches = np.array([patches[:, :, i].ravel() for i in range(3)])
        inpatch_ind = inpatch_ind.ravel()
        cond_means, cond_covs = self._get_cond_params(patches, inpatch_ind)

        samples = []
        for i in range(3):
            samples.append(np.random.multivariate_normal(cond_means[i], cond_covs[i], num_samples))
        samples = np.array(samples).swapaxes(0, 1).swapaxes(1, 2)
        minvals = self.minmax_vals[0].ravel()[win_ind].reshape((self.win_size**2, 3))
        maxvals = self.minmax_vals[1].ravel()[win_ind].reshape((self.win_size**2, 3))
        for i in range(num_samples):
            samples[i][samples[i]<minvals] = minvals[samples[i]<minvals]
            samples[i][samples[i]>maxvals] = maxvals[samples[i]>maxvals]

        return samples

    def _get_surr_patch(self, feat_vector, win_ind):
        height = self.image_size[0]
        width = self.image_size[1]
        patch_ind = np.array(range(self.patch_size**2)).reshape((self.patch_size, self.patch_size))
        win_ind = np.array(np.unravel_index(win_ind, self.image_size)).T

        win_ind_onedim = [i for i in win_ind if i[2]==0]

        top_padding = win_ind_onedim[0][0]
        left_padding = win_ind_onedim[0][1]
        bottom_padding = height - win_ind_onedim[-1][0] - 1
        right_padding = width - win_ind_onedim[-1][1] - 1

        rel_height = top_padding - height + self.patch_size
        rel_width = left_padding - width + self.patch_size

        if top_padding <= self.padding:
            if left_padding <= self.padding:
                patch = feat_vector[:self.patch_size][:, :self.patch_size, :]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[:self.patch_size][:, width - self.patch_size:, :]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[:self.patch_size][:, left_padding -
                                                      self.padding:left_padding - 
                                                      self.padding + self.patch_size, :]
                inpatch_ind = patch_ind[top_padding:top_padding +
                                        self.win_size][:, self.padding:self.padding + self.win_size]
        elif bottom_padding <= self.padding:
            if left_padding <= self.padding:
                patch = feat_vector[height-self.patch_size:][:, :self.patch_size, :]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[height-self.patch_size:][:, width - self.patch_size:, :]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[height-self.patch_size:][:, left_padding -
                                                             self.padding:left_padding -
                                                             self.padding + self.patch_size, :]
                inpatch_ind = patch_ind[rel_height:rel_height +
                                        self.win_size][:, self.padding:self.padding + self.win_size]
        else:
            patch_height = top_padding - self.padding
            patch_width = left_padding - self.padding
            if left_padding <= self.padding:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, :self.patch_size, :]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, left_padding:left_padding + self.win_size]
            elif right_padding <= self.padding:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, width - self.patch_size:, :]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, rel_width:rel_width + self.win_size]
            else:
                patch = feat_vector[patch_height:patch_height +
                                    self.patch_size][:, patch_width:patch_width + self.patch_size, :]
                inpatch_ind = patch_ind[self.padding:self.padding +
                                        self.win_size][:, self.padding:self.padding + self.win_size]

        return patch, inpatch_ind
