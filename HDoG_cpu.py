#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.morphology
import sklearn.mixture


class HDoG_CPU(object):
    def __init__(self, width=2560, height=2160, depth=None, sigma_xy=(4.0, 6.0), sigma_z=(1.8,2.7),
                 radius_small=(24,3), radius_large=(100,5), min_intensity=1000, gamma=1.0):
        self.width  = width
        self.height = height
        self.depth  = depth

        if type(sigma_xy) in [float,int]:
            self.sigma_xy = (sigma_xy, sigma_xy*1.5)
        else:
            self.sigma_xy = sigma_xy
        if type(sigma_z) in [float,int]:
            self.sigma_z = (sigma_z, sigma_z*1.5)
        else:
            self.sigma_z = sigma_z

        if not radius_small:
            self.radius_small_xy = int(self.sigma_xy[1]*4)
            self.radius_small_z = int(self.sigma_z[1]*2)
        else:
            self.radius_small_xy = radius_small[0]
            self.radius_small_z = radius_small[1]
        self.size_small = (self.radius_small_z*2+1, self.radius_small_xy*2+1, self.radius_small_xy*2+1)

        if not radius_large:
            self.radius_large_xy = int(self.sigma_xy[1]*30)
            self.radius_large_xy = int(self.sigma_z[1]*10)
        else:
            self.radius_large_xy = radius_large[0]
            self.radius_large_z = radius_large[1]
        self.size_large = (self.radius_large_z*2+1, self.radius_large_xy*2+1, self.radius_large_xy*2+1)

        self.min_intensity = min_intensity

        self.gamma = gamma
        self.normalizer = (self.sigma_xy[0]**(gamma*2)) * (self.sigma_z[0]**gamma)

    def load_images(self, list_images, dtype=np.uint16):
        imgs = []
        for path in list_images:
            img = np.fromfile(path, dtype=dtype).reshape(self.height, self.width)
            imgs.append(img)
        imgs = np.array(imgs)
        self.depth = imgs.shape[0]
        return imgs

    def Normalize(self, src_img):
        dilation_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_dilation(src_img, size=self.size_large, mode="nearest").astype(np.float32),
            size=self.size_large, mode="constant", cval=0)
        erosion_l_img = scipy.ndimage.filters.uniform_filter(
            scipy.ndimage.morphology.grey_erosion(src_img, size=self.size_large, mode="nearest").astype(np.float32),
            size=self.size_large, mode="constant", cval=0)

        intensity = src_img.astype(np.float32)
        norm_img = (intensity >= self.min_intensity) * intensity / (dilation_l_img - erosion_l_img)
        return norm_img

    def DoGFilter(self, src_img):
        temp1 = scipy.ndimage.filters.gaussian_filter(
            src_img.astype(np.float32),
            sigma=(self.sigma_z[0],self.sigma_xy[0],self.sigma_xy[0]),
            truncate=2.0, mode="constant", cval=0)
        temp2 = scipy.ndimage.filters.gaussian_filter(
            src_img.astype(np.float32),
            sigma=(self.sigma_z[1],self.sigma_xy[1],self.sigma_xy[1]),
            truncate=2.0, mode="constant", cval=0)
        dog_img = (temp1 - temp2) * self.normalizer
        return dog_img

    def HessianPDFilter(self, dog_img):
        Hz,Hy,Hx = np.gradient(dog_img)
        Hzz,Hyz,Hxz = np.gradient(Hz)
        Hyz,Hyy,Hxy = np.gradient(Hy)
        Hxz,Hxy,Hxx = np.gradient(Hx)
        det_img = Hxx*Hyy*Hzz + 2*Hxy*Hyz*Hxz - Hxx*Hyz*Hyz - Hyy*Hxz*Hxz - Hzz*Hxy*Hxy
        pd_img = np.bitwise_and(np.bitwise_and(Hxx < 0, Hxx*Hyy-Hxy*Hxy > 0), det_img < 0)
        hessian_img = np.array([Hxx,Hxy,Hxz,Hyy,Hyz,Hzz])
        return pd_img, hessian_img

    def ScaleResponse(self, scale_img, pd_img):
        response = np.sum(scale_img*pd_img) / np.sum(pd_img)
        return response

    def CCL(self, pd_img):
        labels_img = skimage.morphology.label(pd_img)
        return labels_img

    def RegionalFeatures(self, norm_img, hessian_img, labels_img):
        on_region = np.nonzero(labels_img)
        labels_list = labels_img[on_region]
        num_labels = np.max(labels_list)

        # max intensity
        max_normalized = scipy.ndimage.maximum(norm_img, labels=labels_img, index=range(1, num_labels+1))

        # region size
        ns = np.ones(len(labels_list))
        region_size = np.bincount(labels_list-1, weights=ns)

        # Regional Hessian Eigenvalues
        HT = np.empty((6, num_labels))
        for i in range(6):
            HT[i] = np.bincount(labels_list-1, weights=hessian_img[i][on_region])

        HT_mat = np.array([
            [HT[0],HT[1],HT[2]],
            [HT[1],HT[3],HT[4]],
            [HT[2],HT[4],HT[5]]
        ]).T
        eigenvals = np.linalg.eigvalsh(HT_mat)
        l1,l2,l3 = eigenvals[:,0],eigenvals[:,1], eigenvals[:,2]
        blobness = l3*l3 / (l1*l2) #l3/np.sqrt(l1*l2)
        structureness = l1*l1 + l2*l2 + l3*l3 #np.sqrt()

        # following code is needed if the label is not relabeled as 1,2,3,...
        #label_values = np.array(sorted(np.unique(labels))) # including background(0)
        #mp = np.arange(0,np.max(label_values)+1)
        #mp[label_values] = np.arange(label_values.shape[0])
        #labels_new = mp[labels]
        zgrid,ygrid,xgrid = np.mgrid[0:self.depth, 0:self.height, 0:self.width]
        centroid_x = np.bincount(labels_img.flatten(), weights=xgrid.flatten())[1:] / region_size#np.bincount(labels_img.flatten())[1:]
        centroid_y = np.bincount(labels_img.flatten(), weights=ygrid.flatten())[1:] / region_size#np.bincount(labels_img.flatten())[1:]
        centroid_z = np.bincount(labels_img.flatten(), weights=zgrid.flatten())[1:] / region_size#np.bincount(labels_img.flatten())[1:]

        df = pd.DataFrame({
            "index": pd.Series(np.arange(num_labels)),
            "intensity": pd.Series(max_normalized),
            "size": pd.Series(region_size),
            "blobness":pd.Series(blobness),
            "structureness":pd.Series(structureness),
            "centroid_x": pd.Series(centroid_x),
            "centroid_y": pd.Series(centroid_y),
            "centroid_z": pd.Series(centroid_z),
        })
        return df

    def classify_unsupervised(self, features, i_feature_maximize=2, n_components=3):
        # input
        # features : (num_regions, num_features)
        # i_feature_maximize : the cluster is selected as desired one
        #                      if it has maximum mean value of this feature
        # n_components : number of gaussians
        #
        # output
        # pred : a region is blob if pred = 1, not blob if pred = 0

        vbgmm = sklearn.mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process", n_components=n_components)
        vbgmm.fit(features)
        pred = vbgmm.predict(features)

        i_pred_cluster = np.argmax(vbgmm.means_[:,i_feature_maximize])
        return pred == i_pred_cluster
