#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os, os.path
import tifffile

from HalfBrainImages import HalfBrainImages

dt_local = np.dtype([
    ('local_x', 'f4'), ('local_y', 'f4'), ('local_z', 'f4'),
    ('structureness', 'f4'), ('blobness', 'f4'),('intensity', 'f4'),
    ('size', 'u2'),('padding', 'u2'),
])
dt_global = np.dtype([
    ('merged_x', 'f4'), ('merged_y', 'f4'), ('merged_z', 'f4'),
    ('i_x', 'u1'), ('i_y', 'u1'), # should be < 128
    ('FWRV', 'u1'), # 0 for FW(thetaoff), 1 for RV(thetaon)
])

class HDoGResultStack(object):
    def __init__(self, result_file, imagestack, is_FW, i_xy, scales, image_size, clf=None):
        width,height = image_size
        self.imagestack = imagestack
        self.clf = clf

        if not os.path.exists(result_file):
            self.is_dummy = True
            self.data_local = np.zeros(0, dtype=dt_local)
            self.data_global = np.zeros(0, dtype=dt_global)
        else:
            self.is_dummy = False

            scale_x,scale_y,scale_z = scales
            # load data
            data_local = np.fromfile(result_file, dtype=dt_local)
            if clf is not None:
                # use cells predicted as positive by the classifier
                from HDoG_classifier import get_X_3d
                X = get_X_3d(data_local)
                pred = clf.predict(X)
                data_local = data_local[pred]
            data_global = np.zeros(data_local.shape[0], dtype=dt_global)
            data_global["FWRV"] = 0 if is_FW else 1
            data_global["i_x"] = i_xy[0]
            data_global["i_y"] = i_xy[1]
            data_global["merged_x"] = imagestack.offset_x + data_local["local_x"] * scale_x
            data_global["merged_y"] = imagestack.offset_y + data_local["local_y"] * scale_y
            data_global["merged_z"] = imagestack.offset_z + data_local["local_z"] * scale_z

            # flip
            if scale_x < 0:
                data_global["merged_x"] -= width * scale_x
            if scale_y < 0:
                data_global["merged_y"] -= height * scale_y

            self.data_local = data_local
            self.data_global = data_global

    def get_stack_src_img(self, zlim=None, verbose=True):
        # zlim = (start_z, end_z)
        if verbose: print(self.imagestack.path)
        list_imagefiles = self.imagestack.list_imagefiles_no_dummy
        if verbose: print("\tnumber of images:{}".format(len(list_imagefiles)))
        if zlim:
            if verbose: print("\tspecified images:{}".format(len(list_imagefiles[zlim[0]:zlim[1]])))
            list_imagefiles = list_imagefiles[zlim[0]:zlim[1]]

        imgs = []
        for imgfile in list_imagefiles:
            imgs.append(imgfile.load_image())
        imgs = np.array(imgs)
        return imgs

    def save_stack_src_img(self, dst_path, zlim=None, verbose=True):
        src_img = self.get_stack_src_img(zlim,verbose)
        tifffile.imsave(dst_path, src_img)
        return

    def get_substack_indicator(self, zlim):
        # zlim = (start_z, end_z)
        in_substack = np.bitwise_and(self.data_local["local_z"] >= zlim[0], self.data_local["local_z"] < zlim[1])
        return in_substack

class HalfBrainCells(object):
    def __init__(self, paramfile, is_FW=True, halfbrain_images=None, clf=None):
        print("\n[*] Initializing CellHalfBrain({})".format(paramfile))
        if halfbrain_images is not None:
            assert isinstance(halfbrain_images, HalfBrainImages)
            self.halfbrain_images = halfbrain_images
        else:
            self.halfbrain_images = HalfBrainImages(paramfile)

        self.dict_stacks = {}
        self.clf = clf
        # load result
        result_dir = os.path.join(self.halfbrain_images.params["dst_basedir"])
        if self.clf is None:
            print("<candidate mode>")
        else:
            print("<predicted mode>")

        is_exists_result = np.zeros((len(self.halfbrain_images.list_ynames), len(self.halfbrain_images.list_xnames)), dtype=np.bool)
        width = self.halfbrain_images.params["input_image_info"]["width"]
        height = self.halfbrain_images.params["input_image_info"]["height"]
        total_regions = 0
        list_centroid_xs = []
        list_centroid_ys = []
        list_centroid_zs = []
        for iy,yname in enumerate(self.halfbrain_images.list_ynames):
            for ix,xname in enumerate(self.halfbrain_images.list_xnames):
                result_file = os.path.join(result_dir, "{}_{}.bin".format(yname,xname))
                imagestack = self.halfbrain_images.get_imagestack_by_xyname(xname,yname)
                cellstack = HDoGResultStack(
                    result_file, imagestack, is_FW, (ix,iy),
                    (
                        self.halfbrain_images.scale_x,
                        self.halfbrain_images.scale_y,
                        self.halfbrain_images.scale_z,
                    ), (width, height), self.clf)
                if not cellstack.is_dummy:
                    is_exists_result[iy,ix] = True
                    total_regions += len(cellstack.data_global)

                self.dict_stacks[(xname,yname)] = cellstack
                list_centroid_xs.append(cellstack.data_global["merged_x"])
                list_centroid_ys.append(cellstack.data_global["merged_y"])
                list_centroid_zs.append(cellstack.data_global["merged_z"])
        self.centroid_xs = np.concatenate(list_centroid_xs)
        self.centroid_ys = np.concatenate(list_centroid_ys)
        self.centroid_zs = np.concatenate(list_centroid_zs)

        print("\tResult exists for {} stacks per {} x {}".format(
            np.count_nonzero(is_exists_result),
            len(self.halfbrain_images.list_xnames),
            len(self.halfbrain_images.list_ynames)
        ))
        print("\tTotal Regions: {}".format(total_regions))
        if total_regions > 0:
            print("\tcentroid_x range: {} - {}".format(np.min(self.centroid_xs),np.max(self.centroid_xs)))
            print("\tcentroid_y range: {} - {}".format(np.min(self.centroid_ys),np.max(self.centroid_ys)))
            print("\tcentroid_z range: {} - {}".format(np.min(self.centroid_zs),np.max(self.centroid_zs)))

    def get_stack_by_xyname(self, xname, yname):
        stack = self.dict_stacks.get((xname,yname), None)
        if not stack:
            raise ValueError
        else:
            return stack

    def get_stack(self, pos_xy=None, i_xy=None, verbose=True):
        # specify either `pos_xy` or `i_xy`
        assert pos_xy or i_xy

        stack_xs = self.halfbrain_images.list_offset_xs
        stack_ys = self.halfbrain_images.list_offset_ys
        if not i_xy and pos_xy:
            i_xy = (np.where(np.array(stack_xs) <= pos_xy[0])[0][-1],
                    np.where(np.array(stack_ys) <= pos_xy[1])[0][-1])
        xname = self.halfbrain_images.list_xnames[i_xy[0]]
        yname = self.halfbrain_images.list_ynames[i_xy[1]]
        if verbose:
            print("\txname: {}\tyname:{}".format(xname,yname))

        return self.dict_stacks[(xname,yname)]
