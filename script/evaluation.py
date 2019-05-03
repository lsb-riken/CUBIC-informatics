#!/usr/bin/env python
#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os.path, joblib
import xml.etree.ElementTree as ET
import scipy.ndimage
from scipy.spatial import distance_matrix
import joblib
import tifffile
from skimage.feature import peak_local_max
import scipy.ndimage
import skimage.restoration

from HalfBrainImages import HalfBrainImages
from MergeBrain import WholeBrainCells, WholeBrainImages, dt_scalemerged
from HDoG_classifier import *

dt_centroid = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
])

class Evaluation(object):
    def __init__(self, evalname, wholebrain_cells=None, data_local=None, halfbrain_images=None, dict_channel_wholebrain_images={},
                 flip_local_x=False, flip_local_y=False, is_ave=False, verbose=True):
        """
        evalname is in the following format :
        `REGIONNAME_FWorRV_ZLOCALstart-ZLOCALend_YNAME_XNAME_YLOCALstart-YLOCALend_XLOCALstart_XLOCALend`
        * REGIONNAME is a label for human
        * FWorRV should be either `FW`, `off`, `RV` or `on`
        * YNAME,XNAME specify which stack to be used.
        * ZLOCALstart,ZLOCALend specify which images in the stack to be used. 0 corresponds to the first image in the stack.
        * YLOCALstart,YLOCAL_end,XLOCALstart,XLOCALend specify which area in images to be used. (0,0) corresponds to the top left pixel.

        wholebrain_cells is used for obtaining :
        * data_local from cellstack (centroid coordinates, regional features)
        * data_scalemerged (which cells are valid after merged)
        * image properties (width, height)

        You can specify data_local & halfbrain_images instead of wholebrain_cells.
        If data_local is provided,
        it is used instead the one obtained from wholebrain_cells, and data_scalemerged is not loaded.

        dict_channel_wholebrain_images is a mapping
        from channel name to the corresponding WholeBrainImages instance
        """
        assert ((wholebrain_cells is not None) and isinstance(wholebrain_cells, WholeBrainCells)) \
            or ((data_local is not None) and (halfbrain_images is not None) and isinstance(halfbrain_images, HalfBrainImages))
        for wbi in dict_channel_wholebrain_images.values():
            assert isinstance(wbi, WholeBrainImages)

        region_name,FWRV,zlocal,yname,xname,ylocal,xlocal = evalname.split("_")
        self.region_name = region_name
        self.xname = xname
        self.yname = yname
        self.is_FW = (FWRV == "off") or (FWRV == "FW")
        self.local_xlim = np.array([int(x) for x in xlocal.split("-")], dtype=int)
        self.local_ylim = np.array([int(y) for y in ylocal.split("-")], dtype=int)
        self.local_zlim = np.array([int(z) for z in zlocal.split("-")], dtype=int)
        self.xwidth = self.local_xlim[1]-self.local_xlim[0]
        self.ywidth = self.local_ylim[1]-self.local_ylim[0]
        self.zwidth = self.local_zlim[1]-self.local_zlim[0]
        self.is_ave = is_ave
        self.verbose = verbose

        # flip_local_y is useful to match the ROI between channels
        self.flip_local_x = flip_local_x
        self.flip_local_y = flip_local_y

        if wholebrain_cells is not None:
            self.wbc = wholebrain_cells
            if self.is_FW:
                self.hbc = self.wbc.halfbrain_cells_FW
                self.hbi = self.wbc.wholebrain_images.halfbrain_FW
                dict_channel_hbi = {k:v.halfbrain_FW for k,v in dict_channel_wholebrain_images}
            else:
                self.hbc = self.wbc.halfbrain_cells_RV
                self.hbi = self.wbc.wholebrain_images.halfbrain_RV
                dict_channel_hbi = {k:v.halfbrain_RV for k,v in dict_channel_wholebrain_images}
        else:
            self.wbc = None
            self.hbc = None
            self.hbi = halfbrain_images
            dict_channel_hbi = {}

        self.width = self.hbi.params["input_image_info"]["width"]
        self.height = self.hbi.params["input_image_info"]["height"]
        self.imagestack = self.hbi.get_imagestack_by_xyname(xname=xname, yname=yname)
        self.dict_channel_imagestack = {
            k:v.get_imagestack_by_xyname(xname=xname, yname=yname)
            for k,v in dict_channel_hbi
        }

        if data_local is not None:
            self.data_local = data_local.copy()
            self.data_scalemerged_valid = np.zeros_like(self.data_local, dtype=dt_scalemerged)
            self.data_scalemerged_valid["is_valid"] = True
        else:
            # if data_local is not directly provided, get it from cellstack
            self.cellstack = self.hbc.get_stack_by_xyname(xname=xname, yname=yname)
            # load scalemerged
            if self.is_FW:
                pklpath_scalemerged =  os.path.join(self.wbc.wholebrain_images.params["dst_basedir"], "FW", "{}_{}.pkl".format(yname,xname))
            else:
                pklpath_scalemerged =  os.path.join(self.wbc.wholebrain_images.params["dst_basedir"], "RV", "{}_{}.pkl".format(yname,xname))

            if self.verbose: print("loading", pklpath_scalemerged)
            self.data_scalemerged = joblib.load(pklpath_scalemerged)
            self.data_scalemerged_valid = self.data_scalemerged[self.data_scalemerged["is_valid"]]

            # only valid cells in the stack
            self.data_local = self.cellstack.data_local[self.data_scalemerged["is_valid"]].copy()

        # flip x, y coordinates
        if flip_local_x:
            self.data_local["local_x"] = self.width - self.data_local["local_x"]
        if flip_local_y:
            self.data_local["local_y"] = self.height - self.data_local["local_y"]

        # choosing the cells within the region of interest
        self.in_substack = np.bitwise_and(self.data_local["local_z"] >= self.local_zlim[0], self.data_local["local_z"] <= self.local_zlim[1])
        self.in_ROI = np.bitwise_and(
            self.in_substack,
            np.bitwise_and(
                np.bitwise_and(self.data_local["local_y"] >= self.local_ylim[0], self.data_local["local_y"] <= self.local_ylim[1]),
                np.bitwise_and(self.data_local["local_x"] >= self.local_xlim[0], self.data_local["local_x"] <= self.local_xlim[1])
            )
        )
        self.data_substack = self.data_local[self.in_substack].copy()
        self.data_substack["local_z"] -= self.local_zlim[0]
        self.data_ROI = self.data_local[self.in_ROI].copy()
        self.data_ROI["local_x"] -= self.local_xlim[0]
        self.data_ROI["local_y"] -= self.local_ylim[0]
        self.data_ROI["local_z"] -= self.local_zlim[0]

        self.data_FP = np.empty((0,), dtype=dt_centroid)
        self.data_FN = np.empty((0,), dtype=dt_centroid)

    def get_substack_img(self, channel=None):
        if channel is None:
            imagestack = self.imagestack
        else:
            imagestack = self.dict_channel_imagestack[channel]

        list_imgfile = [imagestack.get_imagefile_by_fname(fname)
                        for fname in imagestack.list_fnames[self.local_zlim[0]:self.local_zlim[1]]]
        substack_img = np.array([
            np.fromfile(imgfile.fullpath, dtype=np.uint16).reshape(self.height,self.width)
            for imgfile in list_imgfile])
        if self.flip_local_x:
            substack_img = substack_img[:,:,::-1]
        if self.flip_local_y:
            substack_img = substack_img[:,::-1,:]
        return substack_img

    def get_ROI_img(self, channel=None):
        substack_img = self.get_substack_img(channel)
        return substack_img[:, self.local_ylim[0]:self.local_ylim[1],
                            self.local_xlim[0]:self.local_xlim[1]]

    def make_pred_img(self, clf=None, data=None, sigma=(1.8, 4.0, 4.0), show_negative=False, normalizer_img=None):
        assert (clf is not None) or (type(data) is np.ndarray)
        if clf:
            X_ROI = get_X_with_normalizer(
                self.data_ROI, self.data_scalemerged_valid[self.in_ROI],
                is_ave=self.is_ave, normalizer_img=normalizer_img)
            pred = clf.predict(X_ROI)
            if show_negative:
                pred = np.bitwise_not(pred)
            data_pred = self.data_ROI[pred]
        else:
            data_pred = data

        hist,_ = np.histogramdd(
            np.vstack([
                data_pred["local_z"],
                data_pred["local_y"],
                data_pred["local_x"],
            ]).T,
            bins=(self.zwidth,self.ywidth,self.xwidth),
            range=[(0,self.zwidth),(0,self.ywidth),(0,self.xwidth)]
        )
        if self.verbose: print(np.count_nonzero(hist))
        pred_img = np.array(hist, dtype=np.uint16) * 5000
        pred_img = scipy.ndimage.filters.gaussian_filter(
            pred_img, sigma=sigma)
        return pred_img

    def make_TP_FP_DC_pred_image(self, clf, sigma=(1.8, 4.0, 4.0), d_th=None, return_data=False, normalizer_img=None):
        X_ROI = get_X_with_normalizer(
            self.data_ROI, self.data_scalemerged_valid[self.in_ROI],
            is_ave=self.is_ave, normalizer_img=normalizer_img)
        pred = clf.predict(X_ROI)
        data_pred = self.data_ROI[pred]

        notin_margin_P = self.notin_margin(self.data_centroids_P)
        notin_margin_test = self.notin_margin(data_pred)
        M = np.count_nonzero(notin_margin_P)
        N = np.count_nonzero(notin_margin_test)
        if self.verbose:
            print("Positive notin margin:", M)
            print("Predicted notin margin:", N)
        distance_P_test = distance_matrix(
            self.physical_scale(data_pred),
            self.physical_scale(self.data_centroids_P), p=2)
        if self.verbose: print("shape of distance_P_test:",distance_P_test.shape)
        distance_P_P = distance_matrix(
            self.physical_scale(self.data_centroids_P),
            self.physical_scale(self.data_centroids_P), p=2)
        distance_P_P[np.diag_indices(self.data_centroids_P.shape[0])] = np.inf
        if self.verbose: print("shape of distance_P_P:",distance_P_P.shape)
        if not d_th:
            d_th = np.mean(np.min(distance_P_P, axis=0))
        if self.verbose: print("d_th:", d_th)

        TP = np.min(distance_P_test[notin_margin_test,:], axis=1) <= d_th
        FP = np.bitwise_not(TP)
        DC = np.empty(np.count_nonzero(TP), dtype=np.bool)
        DC[:] = False
        TP_resp = np.argmin(distance_P_test[notin_margin_test,:][TP,:], axis=1)
        for p in np.unique(TP_resp):
            if np.count_nonzero(TP_resp == p) > 1:
                DC[TP_resp==p] = True
        data_TP = data_pred[notin_margin_test][TP]
        data_FP = data_pred[notin_margin_test][FP]
        data_DC = data_pred[notin_margin_test][TP][DC]
        data_truth = self.data_centroids_P
        if self.verbose:
            print("TP:", data_TP.shape)
            print("FP:", data_FP.shape)
            print("DC:", data_DC.shape)
            print("Truth:", data_truth.shape)

        hist_TP,_ = np.histogramdd(
            np.vstack([
                data_TP["local_z"],
                data_TP["local_y"],
                data_TP["local_x"],
            ]).T,
            bins=(self.zwidth,self.ywidth,self.xwidth),
            range=[(0,self.zwidth),(0,self.ywidth),(0,self.xwidth)]
        )
        hist_FP,_ = np.histogramdd(
            np.vstack([
                data_FP["local_z"],
                data_FP["local_y"],
                data_FP["local_x"],
            ]).T,
            bins=(self.zwidth,self.ywidth,self.xwidth),
            range=[(0,self.zwidth),(0,self.ywidth),(0,self.xwidth)]
        )
        hist_DC,_ = np.histogramdd(
            np.vstack([
                data_DC["local_z"],
                data_DC["local_y"],
                data_DC["local_x"],
            ]).T,
            bins=(self.zwidth,self.ywidth,self.xwidth),
            range=[(0,self.zwidth),(0,self.ywidth),(0,self.xwidth)]
        )
        if self.verbose:
            print(np.count_nonzero(hist_TP))
            print(np.count_nonzero(hist_FP))
            print(np.count_nonzero(hist_DC))
        pred_img_TP = np.array(hist_TP, dtype=np.uint16) * 5000
        pred_img_TP = scipy.ndimage.filters.gaussian_filter(
            pred_img_TP, sigma=sigma)
        pred_img_FP = np.array(hist_FP, dtype=np.uint16) * 5000
        pred_img_FP = scipy.ndimage.filters.gaussian_filter(
            pred_img_FP, sigma=sigma)
        pred_img_DC = np.array(hist_DC, dtype=np.uint16) * 5000
        pred_img_DC = scipy.ndimage.filters.gaussian_filter(
            pred_img_DC, sigma=sigma)
        if return_data:
            return data_TP,data_FP,data_DC,pred_img_TP, pred_img_FP, pred_img_DC
        else:
            return pred_img_TP, pred_img_FP, pred_img_DC

    def plot_slice_image_and_cells(self, z, d=1, clf=None, d_th=None, only_ROI=True, blob_size=2, normalizer_img=None):
        if only_ROI:
            if clf:
                X_ROI = get_X_with_normalizer(
                    self.data_ROI, self.data_scalemerged_valid[self.in_ROI],
                    is_ave=self.is_ave, normalizer_img=normalizer_img)
                pred = clf.predict(X_ROI)
                data_pred = self.data_ROI[pred]
            else:
                data_pred = self.data_ROI
        else:
            if clf:
                X_local = get_X_3d(self.data_local, is_ave=self.is_ave)
                pred = clf.predict(X_local)
                data_pred = self.data_local[pred]
            else:
                data_pred = self.data_local

        # show image
        if only_ROI:
            ROI_img = self.get_ROI_img()
            plt.imshow(ROI_img[z], cmap="gray")
        else:
            substack_img = self.get_substack_img()
            plt.imshow(substack_img[z], cmap="gray")

        # show ROI boundary
        if not only_ROI:
            plt.hlines(self.local_ylim,
                       xmin=self.local_xlim[0],
                       xmax=self.local_xlim[1],
                       color="y", linestyle="dashed", linewidth=1)
            plt.vlines(self.local_xlim,
                       ymin=self.local_ylim[0],
                       ymax=self.local_ylim[1],
                       color="y", linestyle="dashed", linewidth=1)

        if not clf:
            plt.scatter(data_pred["local_x"],
                        data_pred["local_y"], s=blob_size, c="red", alpha=0.5)

        # show false positives, false negatives
        else:
            notin_margin_P = self.notin_margin(self.data_centroids_P)
            notin_margin_test = self.notin_margin(data_pred)
            M = np.count_nonzero(notin_margin_P)
            N = np.count_nonzero(notin_margin_test)
            if self.verbose:
                print("Positive notin margin:", M)
                print("Predicted notin margin:", N)
            distance_P_test = distance_matrix(
                self.physical_scale(data_pred),
                self.physical_scale(self.data_centroids_P), p=2)
            if self.verbose: print("shape of distance_P_test:",distance_P_test.shape)
            distance_P_P = distance_matrix(
                self.physical_scale(self.data_centroids_P),
                self.physical_scale(self.data_centroids_P), p=2)
            distance_P_P[np.diag_indices(self.data_centroids_P.shape[0])] = np.inf
            if self.verbose: print("shape of distance_P_P:",distance_P_P.shape)
            if not d_th:
                d_th = np.mean(np.min(distance_P_P, axis=0))
            if self.verbose: print("d_th:", d_th)

            TP = np.min(distance_P_test[notin_margin_test,:], axis=1) <= d_th
            FP = np.bitwise_not(TP)
            DC = np.empty(np.count_nonzero(TP), dtype=np.bool)
            DC[:] = False
            TP_resp = np.argmin(distance_P_test[notin_margin_test,:][TP,:], axis=1)
            for p in np.unique(TP_resp):
                if np.count_nonzero(TP_resp == p) > 1:
                    DC[TP_resp==p] = True
            data_TP = data_pred[notin_margin_test][TP]
            data_FP = data_pred[notin_margin_test][FP]
            data_DC = data_pred[notin_margin_test][TP][DC]
            if self.verbose:
                print("TP:", data_TP.shape)
                print("FP:", data_FP.shape)
                print("DC:", data_DC.shape)
            data_truth = self.data_centroids_P
            if self.verbose: print("Truth:", data_truth.shape)

            # only slice
            if only_ROI:
                in_slice_TP = np.bitwise_and(data_TP["local_z"] >= z-d,
                                             data_TP["local_z"] <= z+d)
                in_slice_FP = np.bitwise_and(data_FP["local_z"] >= z-d,
                                             data_FP["local_z"] <= z+d)
                in_slice_DC = np.bitwise_and(data_DC["local_z"] >= z-d,
                                             data_DC["local_z"] <= z+d)
                if self.verbose: print(in_slice_DC.shape)
                in_slice_truth = np.bitwise_and(data_truth["local_z"] >= z-d,
                                                data_truth["local_z"] <= z+d)
                data_TP_slice = data_TP[in_slice_TP]
                data_FP_slice = data_FP[in_slice_FP]
                data_DC_slice = data_DC[in_slice_DC]
                data_truth_slice = data_truth[in_slice_truth]
            else:
                in_slice_TP = np.bitwise_and(data_TP["local_z"] >= local_zlim[0]+z-d,
                                             data_TP["local_z"] <= local_zlim[0]+z+d)
                in_slice_FP = np.bitwise_and(data_FP["local_z"] >= local_zlim[0]+z-d,
                                             data_FP["local_z"] <= local_zlim[0]+z+d)
                in_slice_DC = np.bitwise_and(data_DC["local_z"] >= local_zlim[0]+z-d,
                                             data_DC["local_z"] <= local_zlim[0]+z+d)
                in_slice_truth = np.bitwise_and(data_truth["local_z"] >= z-d,
                                                data_truth["local_z"] <= z+d)
                data_TP_slice = data_TP[in_slice_TP]
                data_FP_slice = data_FP[in_slice_FP]
                data_DC_slice = data_DC[in_slice_DC]
                data_truth_slice = data_truth[in_slice_truth]
                data_truth_slice["local_x"] += local_xlim[0]
                data_truth_slice["local_y"] += local_ylim[0]

            plt.scatter(data_TP_slice["local_x"],
                        data_TP_slice["local_y"], s=blob_size, c="red", alpha=0.5)
            plt.scatter(data_FP_slice["local_x"],
                        data_FP_slice["local_y"], s=blob_size, c="pink", alpha=0.5)
            plt.scatter(data_DC_slice["local_x"],
                        data_DC_slice["local_y"], s=blob_size, c="orange", alpha=0.7)
            plt.scatter(data_truth_slice["local_x"],
                        data_truth_slice["local_y"], s=blob_size, c="blue", alpha=0.5)
        return

    def plot_substack_cells(self, clf=None):
        if clf:
            X_substack = get_X_3d(self.data_substack, is_ave=self.is_ave)
            pred = clf.predict(X_substack)
            data_pred = self.data_substack[pred]
        else:
            data_pred = self.data_substack

        plt.scatter(data_pred["local_x"],
                    data_pred["local_y"], s=2)
        plt.hlines(self.local_ylim,
                   xmin=self.local_xlim[0],
                   xmax=self.local_xlim[1],
                  color="k", linestyle="dashed", linewidth=1)
        plt.vlines(self.local_xlim,
                   ymin=self.local_ylim[0],
                   ymax=self.local_ylim[1],
                  color="k", linestyle="dashed", linewidth=1)


    def plot_ROI_cells_xyz(self):
        fig = plt.figure(figsize=(12, 5))
        fig.add_subplot(1,2,1)
        plt.scatter(self.data_ROI["local_x"], self.data_ROI["local_y"])
        if self.data_FN.shape[0] > 0:
            plt.scatter(self.data_FN["local_x"], self.data_FN["local_y"])
        if self.data_FP.shape[0] > 0:
            plt.scatter(self.data_FP["local_x"], self.data_FP["local_y"])
        plt.hlines([10, self.ywidth-10],
                   xmin=10, xmax=self.xwidth-10,
                   color="k", linestyle="dashed", linewidth=1)
        plt.vlines([10, self.xwidth-10],
                   ymin=10, ymax=self.ywidth-10,
                   color="k", linestyle="dashed", linewidth=1)
        fig.add_subplot(1,2,2)
        plt.hist(self.data_ROI["local_z"], bins=self.zwidth, range=(0, self.zwidth))
        if self.data_FN.shape[0] > 0:
            plt.hist(self.data_FN["local_z"], bins=self.zwidth, range=(0, self.zwidth))
        if self.data_FP.shape[0] > 0:
            plt.hist(self.data_FP["local_z"], bins=self.zwidth, range=(0, self.zwidth))
        plt.axvline(5, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(self.zwidth-5, color="k", linestyle="dashed", linewidth=1)

    def notin_margin(self, data_centroids):
        return np.bitwise_and(
            np.bitwise_and(
                np.bitwise_and(data_centroids["local_x"] >= 10, data_centroids["local_x"] < self.xwidth-10),
                np.bitwise_and(data_centroids["local_y"] >= 10, data_centroids["local_y"] < self.ywidth-10),
            ),
            np.bitwise_and(data_centroids["local_z"] > 5, data_centroids["local_z"] < self.zwidth-5)
        )

    def calc_score_true(self, xmlpath, clf=None, data=None, num_channel=2, normalizer_img=None):
        # specify either classifier or data
        # that used in generating dataset for manual counting
        assert (clf is not None) or (type(data) is np.ndarray)
        if clf:
            X_ROI = get_X_with_normalizer(
                self.data_ROI, self.data_scalemerged_valid[self.in_ROI],
                is_ave=self.is_ave, normalizer_img=normalizer_img)
            pred = clf.predict(X_ROI)
            data_pred = self.data_ROI[pred]
        else:
            data_pred = data

        with open(xmlpath) as f:
            root = ET.fromstring(f.read())

        for mtype in root.findall("Marker_Data")[0].findall("Marker_Type"):
            t = mtype.find("Type").text
            if t == "2": # False Positive
                for m in mtype.findall("Marker"):
                    self.data_FP = np.concatenate([self.data_FP,
                        np.array([(int(m.find("MarkerX").text),
                                   int(m.find("MarkerY").text),
                                   np.floor(int(m.find("MarkerZ").text)/num_channel))],
                                 dtype=dt_centroid)])
            elif t == "3": # False Negative
                for m in mtype.findall("Marker"):
                    self.data_FN = np.concatenate([self.data_FN,
                        np.array([(int(m.find("MarkerX").text),
                                   int(m.find("MarkerY").text),
                                   np.floor(int(m.find("MarkerZ").text)/num_channel))],
                                 dtype=dt_centroid)])
                            # exclude marginal ones
        # create TP data
        distance_predicted_FP = distance_matrix(
            self.physical_scale(data_pred), self.physical_scale(self.data_FP), p=2)
        i_FP = np.unique(distance_predicted_FP.argmin(axis=0))
        if self.verbose: print("FP:", self.data_FP.shape, "unique FP:",i_FP)

        self.data_centroids_P = np.zeros(data_pred.shape[0]-i_FP.shape[0]+self.data_FN.shape[0], dtype=dt_centroid)
        self.data_centroids_P["local_x"] = np.concatenate([np.delete(data_pred["local_x"], i_FP), self.data_FN["local_x"]])
        self.data_centroids_P["local_y"] = np.concatenate([np.delete(data_pred["local_y"], i_FP), self.data_FN["local_y"]])
        self.data_centroids_P["local_z"] = np.concatenate([np.delete(data_pred["local_z"], i_FP), self.data_FN["local_z"]])

        # calc metrics
        notin_margin_predicted = self.notin_margin(data_pred)
        notin_margin_FP = self.notin_margin(self.data_FP)
        notin_margin_FN = self.notin_margin(self.data_FN)

        P = np.count_nonzero(notin_margin_predicted)
        FP = np.count_nonzero(notin_margin_FP)
        TP = P - FP
        FN = np.count_nonzero(notin_margin_FN)
        precision = TP / P
        recall = TP / (FN + TP)
        Fscore = 2 * (precision*recall) / (precision + recall)
        density = TP / (self.xwidth-20) / (self.ywidth-20) / (self.zwidth-10)
        if self.verbose:
            print("TP+FP: {}".format(P))
            print("TP: {}".format(TP))
            print("FP: {}".format(FP))
            print("FN: {}".format(FN))
            print("precision: {}".format(precision))
            print("recall: {}".format(recall))
            print("F score: {}".format(Fscore))
            print("Cell Density: {}".format(density))
        return {"TP":TP,"FP":FP,"FN":FN,"precision":precision, "recall":recall, "F":Fscore, "density":density}

    def get_perturbed_true(self):
        # return perturbed true points
        data_centroids = self.data_centroids_P + np.random.normal(size=self.data_centroids_P.shape,
                                                        loc=(0,0,0), scale=(1,1,0.2))
        return data_centroids

    def physical_scale(self, data_centroids):
        data_phy = np.empty((data_centroids.shape[0], 3), dtype=np.float32)
        data_phy[:,0] = data_centroids["local_x"] * self.hbi.scale_x
        data_phy[:,1] = data_centroids["local_y"] * self.hbi.scale_y
        data_phy[:,2] = data_centroids["local_z"] * self.hbi.scale_z
        return data_phy


    def calc_score_test(self, clf=None, data=None, d_th=None, normalizer_img=None):
        # specify either classifier or data
        # that used in generating dataset for manual counting
        assert (clf is not None) or (type(data) is np.ndarray)
        if clf:
            X_ROI = get_X_with_normalizer(
                self.data_ROI, self.data_scalemerged_valid[self.in_ROI],
                is_ave=self.is_ave, normalizer_img=normalizer_img)
            pred = clf.predict(X_ROI)
            data_pred = self.data_ROI[pred]
        else:
            data_pred = data

        notin_margin_P = self.notin_margin(self.data_centroids_P)
        notin_margin_test = self.notin_margin(data_pred)
        M = np.count_nonzero(notin_margin_P)
        N = np.count_nonzero(notin_margin_test)
        if self.verbose:
            print("Positive notin margin:", M)
            print("Predicted notin margin:", N)
        distance_P_test = distance_matrix(
            self.physical_scale(data_pred),
            self.physical_scale(self.data_centroids_P), p=2)
        if self.verbose: print("shape of distance_P_test:",distance_P_test.shape)
        distance_P_P = distance_matrix(
            self.physical_scale(self.data_centroids_P),
            self.physical_scale(self.data_centroids_P), p=2)
        distance_P_P[np.diag_indices(self.data_centroids_P.shape[0])] = np.inf
        if self.verbose: print("shape of distance_P_P:",distance_P_P.shape)
        if not d_th:
            d_th = np.mean(np.min(distance_P_P, axis=0))
        if self.verbose: print("d_th:", d_th)
        TP = min(np.count_nonzero(np.min(distance_P_test[:,notin_margin_P], axis=0) <= d_th),
                np.count_nonzero(np.min(distance_P_test[notin_margin_test,:], axis=1) <= d_th))
        FP = N - TP
        FN = M - TP
        precision = TP / N
        recall = TP / M
        Fscore = 2 * (precision*recall) / (precision+recall)
        if self.verbose:
            print("TP: {}".format(TP))
            print("FP: {}".format(FP))
            print("FN: {}".format(FN))
            print("precision: {}".format(precision))
            print("recall: {}".format(recall))
            print("F score: {}".format(Fscore))
        d_ths = np.linspace(0,40,20)
        TPs = np.array([min(
            np.count_nonzero(np.min(distance_P_test[:,notin_margin_P], axis=0) <= d),
            np.count_nonzero(np.min(distance_P_test[notin_margin_test,:], axis=1) <= d)) for d in d_ths])
        precisions = TPs / N
        recalls = TPs / M
        Fscores = 2 * (precisions*recalls) / (precisions+recalls)
        fig = plt.figure(figsize=(10,5))
        fig.add_subplot(1,2,1)
        plt.plot(d_ths, precisions, label="precision")
        plt.plot(d_ths, recalls, label="recall")
        plt.plot(d_ths, Fscores, label="F score")
        plt.legend()
        fig.add_subplot(1,2,2)
        plt.scatter(self.data_centroids_P["local_x"], self.data_centroids_P["local_y"], s=2)
        plt.scatter(data_pred["local_x"], data_pred["local_y"], s=2)
        plt.hlines([10, self.ywidth-10],
                   xmin=10, xmax=self.xwidth-10,
                   color="k", linestyle="dashed", linewidth=1)
        plt.vlines([10, self.xwidth-10],
                   ymin=10, ymax=self.ywidth-10,
                   color="k", linestyle="dashed", linewidth=1)
        return {"TP":TP,"FP":FP,"FN":FN,"precision":precision, "recall":recall, "F":Fscore}

