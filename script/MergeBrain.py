#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Downscale images & cells for altas mapping

Usage:
  MergeBrain.py images PARAM_FILE [-p NUM_CPUS] [--exec <path>]
  MergeBrain.py cells PARAM_FILE
  MergeBrain.py full PARAM_FILE [-p NUM_CPUS] [--exec <path>]

Options:
  -h --help        Show this screen.
  --version        Show version.
  -p NUM_CPUS      Number of cpus to be used [default: -1](all available).
  --exec <path>    Location of the executable [default: ./build/ScaleMerge]
"""

import json, glob, os.path, shutil
import tifffile
import functools
from docopt import docopt
import joblib
import subprocess as sp
import pandas as pd
import numpy as np

from HalfBrainCells import HalfBrainCells
from HalfBrainImages import HalfBrainImages


dt_scalemerged = np.dtype([
    ('scaled_x','f4'), ('scaled_y', 'f4'), ('scaled_z', 'f4'),
    ('is_valid', 'bool'),
])

def run_ScaleMerge(paramfile, mergedfile, path_exec, logfile=None, print_output=True):
    mergedfile_mean,mergedfile_max,mergedfile_min = mergedfile
    cmd = " ".join([path_exec, paramfile,
                    mergedfile_mean,mergedfile_max,mergedfile_min])
    print("[*] Executing : {}".format(cmd))
    out = sp.check_output([path_exec, paramfile,
                           mergedfile_mean,mergedfile_max,mergedfile_min])
    if logfile:
        with open(logfile, "wb") as f:
            f.write(out)
    else:
        if print_output:
            print(out.decode())
    return

class WholeBrainImages(object):
    def __init__(self, paramfile, ):
        print("\n[*] Initializing WholeBrain({})".format(paramfile))
        with open(paramfile) as f:
            self.params = json.load(f)

        self.halfbrain_FW = HalfBrainImages(self.params["HDoG_paramfile"]["FW"])
        self.halfbrain_RV = HalfBrainImages(self.params["HDoG_paramfile"]["RV"])

        # asuume scale is equivalent for FW & RV except for direction
        assert abs(self.halfbrain_FW.scale_xy) == abs(self.halfbrain_RV.scale_xy)
        assert abs(self.halfbrain_FW.scale_z) == abs(self.halfbrain_RV.scale_z)
        self.fnames_FW = self.halfbrain_FW.list_fnames_all
        self.fnames_RV = self.halfbrain_RV.list_fnames_all
        self.zs_FW = self.halfbrain_FW.list_zs_all
        self.zs_RV = self.halfbrain_RV.list_zs_all
        self.zs_global_FW = self.halfbrain_FW.list_zs_global_all
        self.zs_global_RV = self.halfbrain_RV.list_zs_global_all

        # boundary position
        fname_boundary_FW = self.params["merge_info"]["boundary_fname"]["FW"]
        fname_boundary_RV = self.params["merge_info"]["boundary_fname"]["RV"]
        if len(self.zs_FW) > 0:
            self.iz_FW_boundary = self.zs_FW.index(int(fname_boundary_FW))
        else:
            self.iz_FW_boundary = 0
        if len(self.zs_RV) > 0:
            self.iz_RV_boundary = self.zs_RV.index(int(fname_boundary_RV))
        else:
            self.iz_RV_boundary = 0

        print("\t boundary for FW ({}) at i={}".format(fname_boundary_FW, self.iz_FW_boundary))
        print("\t boundary for RV ({}) at i={}".format(fname_boundary_RV, self.iz_RV_boundary))

        self.skip_z_FW = 1
        self.skip_z_RV = 1
        self.param_header_FW = ""
        self.param_header_RV = ""
        self.precompute_param_header(is_FW=True)
        self.precompute_param_header(is_FW=False)
        self.bound_z_global_FW = (-np.inf, +np.inf)
        self.bound_z_global_RV = (-np.inf, +np.inf)
        self.merged_depth = None

        self.single_mergedfile_mean = os.path.join(self.params["dst_basedir"], "whole.tif")
        self.single_mergedfile_max = os.path.join(self.params["dst_basedir"], "whole_max.tif")
        self.single_mergedfile_min = os.path.join(self.params["dst_basedir"], "whole_min.tif")

    def precompute_param_header(self, is_FW):
        if is_FW:
            print("[*] Precomputng param header for FW")
            halfbrain = self.halfbrain_FW
            flip_rot_before_info =  self.params["merge_info"]["flip_rot"]["FW"]
        else:
            print("[*] Precomputng param header for RV")
            halfbrain = self.halfbrain_RV
            flip_rot_before_info =  self.params["merge_info"]["flip_rot"]["RV"]

        input_image_info = halfbrain.params["input_image_info"]
        flip_rot_after_info = self.params["scale_info"]["flip_rot"]

        # downscale ratio
        down_scale_xyz = self.params["scale_info"]["downscale_unit"]
        downscale_ratio_xy = float(abs(halfbrain.scale_xy)) / down_scale_xyz # [um / um] = dimensionless
        assert down_scale_xyz % halfbrain.scale_z == 0
        downscale_ratio_z = float(abs(halfbrain.scale_z)) / down_scale_xyz # [um / um] = dimensionless
        skip_z = int(down_scale_xyz / abs(halfbrain.scale_z))
        print("\t downscale ratio for xy : {}".format(downscale_ratio_xy))
        print("\t downscale ratio for z : {} (skip={})".format(downscale_ratio_z, skip_z))

        flip_rot_before = 0
        flip_rot_before += 1 if flip_rot_before_info["flipX"] else 0
        flip_rot_before += 2 if flip_rot_before_info["flipY"] else 0
        flip_rot_before += 4 if flip_rot_before_info["rotCCW"] else 0
        flip_rot_before += 8 if flip_rot_before_info["rotCW"] else 0
        flip_rot_after = 0
        flip_rot_after += 1 if flip_rot_after_info["flipX"] else 0
        flip_rot_after += 2 if flip_rot_after_info["flipY"] else 0
        if flip_rot_before_info["rotCCW"] or flip_rot_before_info["rotCW"]:
            width_loaded = input_image_info["height"]
            height_loaded = input_image_info["width"]
        else:
            width_loaded = input_image_info["width"]
            height_loaded = input_image_info["height"]
        num_xnames = len(halfbrain.list_xnames)
        num_ynames = len(halfbrain.list_ynames)
        param_dict = {
            "width": width_loaded,
            "height": height_loaded,
            "num_xnames": num_xnames,
            "num_ynames": num_ynames,
            "downscale_ratio_xy": downscale_ratio_xy,
            "downscale_ratio_z": downscale_ratio_z,
            "overlap_left": input_image_info["left_margin"],
            "overlap_right": input_image_info["right_margin"],
            "overlap_top": input_image_info["top_margin"],
            "overlap_bottom": input_image_info["bottom_margin"],
            "flip_rot_before": flip_rot_before,
            "flip_rot_after": flip_rot_after,
            "imgformat": 1, # bin
            "showgrid": 0, # no grid
        }

        # compute ScaleMerged parameters for cell coordinate transformation
        # apply transformation as in ScaleMerge
        strip_width = input_image_info["width"] - input_image_info["left_margin"] - input_image_info["right_margin"]
        strip_height = input_image_info["height"] - input_image_info["top_margin"] - input_image_info["bottom_margin"]
        if flip_rot_before_info["rotCCW"] or flip_rot_before_info["rotCW"]:
            strip_width,strip_height = strip_height,strip_width
        # max int less than or equal strip_width * downscale_ratio_xy
        sampled_width = int(strip_width * downscale_ratio_xy)
        sampled_height = int(strip_height * downscale_ratio_xy)
        actual_downscale_ratio_x = sampled_width / strip_width # [pixel / pixel] = dimensionless
        actual_downscale_ratio_y = sampled_height / strip_height # [pixel / pixel] = dimensionless
        kernel_width = strip_width / sampled_width
        kernel_height = strip_height / sampled_height
        merged_width = sampled_width * num_xnames
        merged_height = sampled_height * num_ynames

        margin_left = input_image_info["left_margin"] * actual_downscale_ratio_x
        margin_right = input_image_info["right_margin"] * actual_downscale_ratio_x
        margin_top = input_image_info["top_margin"] * actual_downscale_ratio_y
        margin_bottom = input_image_info["bottom_margin"] * actual_downscale_ratio_y
        if flip_rot_before_info["flipX"]:
            margin_left,margin_right = margin_right,margin_left
        if flip_rot_before_info["flipY"]:
            margin_top,margin_bottom = margin_bottom,margin_top
        if flip_rot_before_info["rotCCW"]:
            margin_left,margin_top,margin_right,margin_bottom = margin_top,margin_right,margin_bottom,margin_left
        if flip_rot_before_info["rotCW"]:
            margin_left,margin_top,margin_right,margin_bottom = margin_bottom,margin_left,margin_top,margin_right
        if flip_rot_after_info["flipX"]:
            margin_left,margin_right = margin_right,margin_left
        if flip_rot_after_info["flipY"]:
            margin_top,margin_bottom = margin_bottom,margin_top
        print("\t original: {} x {} x ({} x {})".format(input_image_info["width"], input_image_info["height"], num_xnames, num_ynames))
        print("\t strip: {} x {} x ({} x {})".format(strip_width, strip_height, num_xnames, num_ynames))
        print("\t sampled: {} x {} x ({} x {})".format(sampled_width, sampled_height, num_xnames, num_ynames))
        print("\t merged: {} x {}".format(merged_width, merged_height))
        print("\t actual downscale ratio : {:.7f} x {:.7f}".format(actual_downscale_ratio_x, actual_downscale_ratio_y))
        print("\t merged_mergin: L:{:.3f} R:{:.3f} T:{:.3f} B:{:.3f}".format(margin_left,margin_right,margin_top, margin_bottom))
        param_dict.update({
            "merged_margin_left": margin_left,
            "merged_margin_right": margin_right,
            "merged_margin_top": margin_top,
            "merged_margin_bottom": margin_bottom,
            "strip_width": strip_width,
            "strip_height": strip_height,
            "sampled_width": sampled_width,
            "sampled_height": sampled_height,
            "actual_downscale_ratio_x": actual_downscale_ratio_x,
            "actual_downscale_ratio_y": actual_downscale_ratio_y,
            "kernel_width": kernel_width,
            "kernel_height": kernel_height,
            "merged_width": merged_width,
            "merged_height": merged_height,
        })
        if is_FW:
            self.skip_z_FW = skip_z
            self.param_scalemerge_FW = param_dict
        else:
            self.skip_z_RV = skip_z
            self.param_scalemerge_RV = param_dict
        return

    def scalemerge(self, num_cpus=-1, dry_run=False, path_exec="./ScaleMerge"):
        print("[*] Starting scalemerge...")
        # Let's start merging FW & RV using boundary information
        scale_z_FW = self.halfbrain_FW.scale_z
        scale_z_RV = self.halfbrain_RV.scale_z
        if self.params["merge_info"]["use_at_boundary"] == "FW":
            use_FW_at_boundary = True
        elif self.params["merge_info"]["use_at_boundary"] == "RV":
            use_FW_at_boundary = False
        else:
            raise TypeError

        print("\t FW length: {}".format(len(self.fnames_FW)))
        print("\t RV length: {}".format(len(self.fnames_RV)))
        indices_FW = range(len(self.fnames_FW))
        indices_RV = range(len(self.fnames_RV))
        zflip = self.params["scale_info"]["flip_rot"]["flipZ"]
        print("\t z flip: {}".format("on" if zflip else "off"))

        is_halfsize = False
        if len(self.zs_global_FW) > 0:
            zs_global_FW0 = self.zs_global_FW[0]
            zs_global_FW1 = self.zs_global_FW[-1]
        else:
            zs_global_FW0 = None
            zs_global_FW1 = None
            is_halfsize = True
        if len(self.zs_global_RV) > 0:
            zs_global_RV0 = self.zs_global_RV[0]
            zs_global_RV1 = self.zs_global_RV[-1]
        else:
            zs_global_RV0 = None
            zs_global_RV1 = None
            is_halfsize = True

        if scale_z_FW * scale_z_RV > 0:
            print("[Case 1-4]")
            print("\t scale_z_FW", scale_z_FW)
            print("\t zs_global_FW[0]:", self.zs_global_FW0)
            print("\t zs_global_FW[-1]:", self.zs_global_FW1)
            print("\t zs_global_RV[0]:", self.zs_global_RV0)
            print("\t zs_global_RV[-1]:", self.zs_global_RV1)
            # suppose FW & RV is growing in the same direction,
            # there is 4 scenarios for merging.
            if scale_z_FW > 0 and (is_halfsize or self.zs_global_FW[0] < self.zs_global_RV[0]):
                print("->[Case 1]")
                # [case 1]
                #  merged: |-FW-->|--RV-->
                #  FW:     |-FW---->
                #  RV:           |---RV-->
                # if halfsize, case2 and case1 comes to the same
                indices_FW_strip = indices_FW[:self.iz_FW_boundary+1][::-1][::self.skip_z_FW][::-1]
                indices_RV_strip = indices_RV[self.iz_RV_boundary:][::self.skip_z_RV]

                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[1:]
                else:
                    indices_FW_strip = indices_FW_strip[:-1]
                is_FWs = [True for _ in indices_FW_strip] + [False for _ in indices_RV_strip]
                merging_fnames = [self.fnames_FW[i] for i in indices_FW_strip] + [self.fnames_RV[i] for i in indices_RV_strip]

            elif scale_z_FW > 0 and self.zs_global_RV[0] < self.zs_global_FW[0]:
                print("->[Case 2]")
                # [case 2]
                #  mergped: |-RV-->|--FW-->
                #  FW:           |---FW-->
                #  RV:     |-RV---->
                indices_RV_strip = indices_RV[:self.iz_RV_boundary+1][::-1][::self.skip_z_RV][::-1]
                indices_FW_strip = indices_FW[self.iz_FW_boundary:][::self.skip_z_FW]
                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[:-1]
                else:
                    indices_FW_strip = indices_FW_strip[1:]
                is_FWs = [False for _ in indices_RV_strip] + [True for _ in indices_FW_strip]
                merging_fnames = [self.fnames_RV[i] for i in indices_RV_strip] + [self.fnames_FW[i] for i in indices_FW_strip]

            elif scale_z_FW < 0  and (is_halfsize or self.zs_global_FW[0] < self.zs_global_RV[0]):
                print("->[Case 3]")
                # [case 3] (reverse case 1)
                #  merged: |-FW-->|--RV-->
                #  FW:     <-FW----|
                #  RV:           <---RV--|
                # if halfsize, case3 and case4 comes to the same
                indices_FW_strip = indices_FW[self.iz_FW_boundary:][::self.skip_z_FW][::-1]
                indices_RV_strip = indices_RV[:self.iz_RV_boundary+1][::-1][::self.skip_z_RV]

                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[1:]
                else:
                    indices_FW_strip = indices_FW_strip[:-1]
                is_FWs = [True for _ in indices_FW_strip] + [False for _ in indices_RV_strip]
                merging_fnames = [self.fnames_FW[i] for i in indices_FW_strip] + [self.fnames_RV[i] for i in indices_RV_strip]

            elif scale_z_FW < 0  and self.zs_global_RV[0] < self.zs_global_FW[0]:
                print("->[Case 4]")
                # [case 4] : reverse case2
                #  mergped: |-RV-->|--FW-->
                #  FW:            <---FW--|
                #  RV:      <-RV----|
                indices_RV_strip = indices_RV[self.iz_RV_boundary:][::self.skip_z_RV][::-1]
                indices_FW_strip = indices_FW[:self.iz_FW_boundary+1][::-1][::self.skip_z_FW]
                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[:-1]
                else:
                    indices_FW_strip = indices_FW_strip[1:]
                is_FWs = [False for _ in indices_RV_strip] + [True for _ in indices_FW_strip]
                merging_fnames = [self.fnames_RV[i] for i in indices_RV_strip] + [self.fnames_FW[i] for i in indices_FW_strip]

            else:
                raise TypeError
        elif scale_z_FW * scale_z_RV < 0:
            # suppose FW & RV is growing in the opposite direction,
            # there is 4 scenarios
            print("[Case 5-8]")
            print("\t scale_z_FW", scale_z_FW)
            print("\t zs_global_FW[0]:", self.zs_global_FW0)
            print("\t zs_global_FW[-1]:", self.zs_global_FW1)
            print("\t zs_global_RV[0]:", self.zs_global_RV0)
            print("\t zs_global_RV[-1]:", self.zs_global_RV1)
            if scale_z_FW < 0 and (is_halfsize or self.zs_global_FW[-1] < self.zs_global_RV[0]):
                print("->[Case 5]")
                # [case 5]
                #  merged: |-FW-->|--RV-->
                #  FW:     <-FW----|
                #  RV:           |---RV-->
                # if halfsize, case5 and case6 comes to the same
                indices_FW_strip = indices_FW[self.iz_FW_boundary:][::self.skip_z_FW][::-1]
                indices_RV_strip = indices_RV[self.iz_RV_boundary:][::self.skip_z_RV]
                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[1:]
                else:
                    indices_FW_strip = indices_FW_strip[:-1]
                is_FWs = [True for _ in indices_FW_strip] + [False for _ in indices_RV_strip]
                merging_fnames = [self.fnames_FW[i] for i in indices_FW_strip] + [self.fnames_RV[i] for i in indices_RV_strip]

            elif scale_z_FW > 0 and (is_halfsize or self.zs_global_FW[-1] > self.zs_global_RV[0]):
                print("->[Case 6]")
                # [case 6]
                #  merged: |-RV-->|--FW-->
                #  FW:           |---FW-->
                #  RV:     <-RV----|
                indices_RV_strip = indices_RV[self.iz_RV_boundary:][::self.skip_z_RV][::-1]
                indices_FW_strip = indices_FW[self.iz_FW_boundary:][::self.skip_z_FW]
                if use_FW_at_boundary:
                    indices_RV_strip = indices_RV_strip[:-1]
                else:
                    indices_FW_strip = indices_FW_strip[1:]
                is_FWs = [False for _ in indices_RV_strip] + [True for _ in indices_FW_strip]
                merging_fnames = [self.fnames_RV[i] for i in indices_RV_strip] + [self.fnames_FW[i] for i in indices_FW_strip]

            elif scale_z_FW > 0 and self.zs_global_FW[-1] < self.zs_global_RV[0]:
                print("->[Case 7]")
                # [case 7] : reverse case5
                raise NotImplementedError
            elif scale_z_FW < 0 and self.zs_global_FW[-1] > self.zs_global_RV[0]:
                print("->[Case 8]")
                # [case 8] : reverse case6
                raise NotImplementedError
            else:
                raise TypeError
        else:
            raise TypeError

        # save boundary point for picking valid cell candidates
        if is_FWs[0]:
            self.bound_z_global_FW = (-np.inf, self.zs_global_FW[self.iz_FW_boundary])
            self.bound_z_global_RV = (self.zs_global_RV[self.iz_RV_boundary], +np.inf)
        else:
            self.bound_z_global_RV = (-np.inf, self.zs_global_RV[self.iz_RV_boundary])
            self.bound_z_global_FW = (self.zs_global_FW[self.iz_FW_boundary], +np.inf)

        self.merged_depth = len(merging_fnames)
        print("\tmerged depth: {}".format(self.merged_depth))
        if is_FWs[0]:
            self.new_origin_z_global = self.zs_global_FW[indices_FW_strip[0]]
        else:
            self.new_origin_z_global = self.zs_global_RV[indices_RV_strip[0]]
        print("\tnew z_global origin : {}".format(self.new_origin_z_global))

        if zflip:
            is_FWs = is_FWs[::-1]
            merging_fnames = merging_fnames[::-1]

        # write paramfiles for each process of ScaleMerge
        total_z_merged = len(merging_fnames)
        mergedfile_mean_basedir = os.path.join(self.params["dst_basedir"], "zs_mean")
        mergedfile_max_basedir = os.path.join(self.params["dst_basedir"], "zs_max")
        mergedfile_min_basedir = os.path.join(self.params["dst_basedir"], "zs_min")
        if not os.path.exists(mergedfile_mean_basedir):
            os.makedirs(mergedfile_mean_basedir)
        if not os.path.exists(mergedfile_max_basedir):
            os.makedirs(mergedfile_max_basedir)
        if not os.path.exists(mergedfile_min_basedir):
            os.makedirs(mergedfile_min_basedir)
        mergedfile_mean_basename = os.path.join(mergedfile_mean_basedir, "{i:04d}.tif")
        mergedfile_max_basename = os.path.join(mergedfile_max_basedir, "{i:04d}.tif")
        mergedfile_min_basename = os.path.join(mergedfile_min_basedir, "{i:04d}.tif")
        mergedfiles = [(
            mergedfile_mean_basename.format(i=i),
            mergedfile_max_basename.format(i=i),
            mergedfile_min_basename.format(i=i),
        )for i in range(total_z_merged)]

        paramfiles = [self.write_paramfile(i,is_FW,merging_fname)
                      for i,(is_FW,merging_fname) in enumerate(zip(is_FWs, merging_fnames))]
        if not dry_run:
            joblib.Parallel(n_jobs=num_cpus, verbose=10)([
                joblib.delayed(run_ScaleMerge)(paramfile,mergedfile, path_exec, print_output=False)
                for paramfile, mergedfile in zip(paramfiles,mergedfiles)
            ])
            print("[*] Concatenating tiff images to single tiff({})".format(self.single_mergedfile_mean))
            img_mergedsingle_mean = np.empty((len(mergedfiles), self.param_scalemerge_FW["merged_height"], self.param_scalemerge_FW["merged_width"]), dtype=np.uint16)
            img_mergedsingle_max = np.empty_like(img_mergedsingle_mean)
            img_mergedsingle_min = np.empty_like(img_mergedsingle_mean)

            for i,(mergedfile_mean,mergedfile_max,mergedfile_min) in enumerate(mergedfiles):
                img_mergedsingle_mean[i,:,:] = tifffile.imread(mergedfile_mean)
                img_mergedsingle_max[i,:,:] = tifffile.imread(mergedfile_max)
                img_mergedsingle_min[i,:,:] = tifffile.imread(mergedfile_min)
            tifffile.imsave(self.single_mergedfile_mean, img_mergedsingle_mean)
            tifffile.imsave(self.single_mergedfile_max, img_mergedsingle_max)
            tifffile.imsave(self.single_mergedfile_min, img_mergedsingle_min)

            print("[*] Deleting temporary tiff images")
            shutil.rmtree(mergedfile_mean_basedir)
            shutil.rmtree(mergedfile_max_basedir)
            shutil.rmtree(mergedfile_min_basedir)
        else:
            print("[*] Skipping ScaleMerge for images")

        for paramfile in paramfiles:
            os.remove(paramfile)
        return

    def write_paramfile(self, i, is_FW, merging_fname):
        paramfile = "/tmp/param_merge_{randomID}_{i:04d}.txt".format(randomID = np.random.randint(2**31), i=i)
        if is_FW:
            param_dict = self.param_scalemerge_FW
            halfbrain = self.halfbrain_FW
        else:
            param_dict = self.param_scalemerge_RV
            halfbrain = self.halfbrain_RV
        param_text = "{width}:{height}:{num_xnames}:{num_ynames}:{downscale_ratio_xy}:{overlap_left}:{overlap_right}:{overlap_top}:{overlap_bottom}:{flip_rot_before}:{flip_rot_after}:{imgformat}:{showgrid}\n".format(**param_dict)

        for yname in halfbrain.list_ynames:
            for xname in halfbrain.list_xnames:
                imagestack = halfbrain.get_imagestack_by_xyname(xname,yname)
                img = imagestack.get_imagefile_by_fname(merging_fname)
                fullpath = img.fullpath if not img.is_dummy else ""
                param_text += fullpath + "\n"

        with open(paramfile, "w") as f:
            f.write(param_text)

        return paramfile

class WholeBrainCells(object):
    def __init__(self, paramfile, wholebrain_images=None, clf=None):
        if wholebrain_images:
            self.wholebrain_images = wholebrain_images
        else:
            self.wholebrain_images = WholeBrainImages(paramfile)

        self.halfbrain_cells_FW = HalfBrainCells(
            self.wholebrain_images.params["HDoG_paramfile"]["FW"],
            is_FW = True,
            halfbrain_images=self.wholebrain_images.halfbrain_FW,
            clf=clf
        )
        self.halfbrain_cells_RV = HalfBrainCells(
            self.wholebrain_images.params["HDoG_paramfile"]["RV"],
            is_FW = False,
            halfbrain_images=self.wholebrain_images.halfbrain_RV,
            clf=clf
        )
        # average mode or not (default: false)
        is_ave_FW = self.halfbrain_cells_FW.halfbrain_images.params["HDoG_param"].get("is_ave_mode", False)
        is_ave_RV = self.halfbrain_cells_RV.halfbrain_images.params["HDoG_param"].get("is_ave_mode", False)
        assert is_ave_FW == is_ave_RV
        self.is_ave = is_ave_FW

    def scalemerge(self):
        # should be called after scalemerge()
        print("[*] Starting scalemerge for HDoG result...")
        cellstacks_FW = self.halfbrain_cells_FW.dict_stacks
        cellstacks_RV = self.halfbrain_cells_RV.dict_stacks
        param_scalemerge_FW = self.wholebrain_images.param_scalemerge_FW
        param_scalemerge_RV = self.wholebrain_images.param_scalemerge_RV
        # scale and merge
        org_scale_xy_FW = float(abs(self.wholebrain_images.halfbrain_FW.scale_xy))
        org_scale_z_FW = float(abs(self.wholebrain_images.halfbrain_FW.scale_z))
        org_scale_xy_RV = float(abs(self.wholebrain_images.halfbrain_RV.scale_xy))
        org_scale_z_RV = float(abs(self.wholebrain_images.halfbrain_RV.scale_z))
        offset_x_FW = self.wholebrain_images.halfbrain_FW.list_offset_xs[0]
        offset_y_FW = self.wholebrain_images.halfbrain_FW.list_offset_ys[0]
        offset_x_RV = self.wholebrain_images.halfbrain_RV.list_offset_xs[0]
        offset_y_RV = self.wholebrain_images.halfbrain_RV.list_offset_ys[0]
        print("\t offset_FW: {},{},{}".format(offset_x_FW,offset_y_FW,self.wholebrain_images.new_origin_z_global))
        print("\t offset_RV: {},{},{}".format(offset_x_RV,offset_y_RV,self.wholebrain_images.new_origin_z_global))
        # flip rot after
        flip_rot_after_info =  self.wholebrain_images.params["scale_info"]["flip_rot"]
        A_FW = np.zeros((3,3))
        A_FW[:2,:2] = np.array(self.wholebrain_images.halfbrain_FW.params["coordinate_info"]["affine_global"])[:2,:2]
        A_FW[2,2] = 1.
        A_FW[np.nonzero(A_FW)] = 1.
        b_FW = np.zeros(3)
        A_RV = np.zeros((3,3))
        A_RV[:2,:2] = np.array(self.wholebrain_images.halfbrain_RV.params["coordinate_info"]["affine_global"])[:2,:2]
        A_RV[2,2] = 1.
        A_RV[np.nonzero(A_RV)] = 1.
        b_RV = np.zeros(3)
        if flip_rot_after_info["flipX"]:
            b_FW[0] += param_scalemerge_FW["merged_width"]
            A_FW[0,:] *= -1
            b_RV[0] += param_scalemerge_RV["merged_width"]
            A_RV[0,:] *= -1
        if flip_rot_after_info["flipY"]:
            b_FW[1] += param_scalemerge_FW["merged_height"]
            A_FW[1,:] *= -1
            b_RV[1] += param_scalemerge_RV["merged_height"]
            A_RV[1,:] *= -1
        if flip_rot_after_info["flipZ"]:
            b_FW[2] += self.wholebrain_images.merged_depth
            A_FW[2,:] *= -1
            b_RV[2] += self.wholebrain_images.merged_depth
            A_RV[2,:] *= -1

        def process_stack(dst_path, cellstack, bound_z, margin_left, margin_top,
                          offset_x, offset_y, offset_z, coeff_x, coeff_y, coeff_z, A, b):
            print("[*] Dumping merged data to {}".format(dst_path))
            if bound_z[0] > bound_z[1]:
                smallest_z,largest_z = bound_z[1],bound_z[0]
            else:
                smallest_z,largest_z = bound_z
            data_scaled = np.zeros(cellstack.data_global.shape[0], dtype=dt_scalemerged)
            data_scaled["is_valid"] = np.bitwise_and(
                smallest_z <= cellstack.data_global["merged_z"],
                cellstack.data_global["merged_z"] <= largest_z)
            #print("\tz_range 1: {} - {}".format(data_valid["centroid_z"].min(), data_valid["centroid_z"].max()))
            centroid_scaled = np.zeros((cellstack.data_global.shape[0],3), dtype=np.float32)
            centroid_scaled[:,0] = (cellstack.data_global["merged_x"] - offset_x) * coeff_x - margin_left
            centroid_scaled[:,1] = (cellstack.data_global["merged_y"] - offset_y) * coeff_y - margin_top
            centroid_scaled[:,2] = (cellstack.data_global["merged_z"] - offset_z) * coeff_z
            #print("\tz_range 2: {} - {}".format(centroid_scaled[:,2].min(), centroid_scaled[:,2].max()))
            centroid_fliprot = A.dot(centroid_scaled.T).T + b
            data_scaled["scaled_x"] = centroid_fliprot[:,0]
            data_scaled["scaled_y"] = centroid_fliprot[:,1]
            data_scaled["scaled_z"] = centroid_fliprot[:,2]
            #print("\tz_range 3: {} - {}".format(data_valid["centroid_z"].min(), data_valid["centroid_z"].max()))
            joblib.dump(data_scaled, dst_path, compress=3)
            return np.count_nonzero(data_scaled["is_valid"])

        dst_basedir = self.wholebrain_images.params["dst_basedir"]
        dst_basedir_FW = os.path.join(dst_basedir,"FW")
        dst_basedir_RV = os.path.join(dst_basedir,"RV")
        if not os.path.exists(dst_basedir_FW):
            os.makedirs(dst_basedir_FW)
        if not os.path.exists(dst_basedir_RV):
            os.makedirs(dst_basedir_RV)
        # Note: parallelizable loop
        dict_num_cells = {}
        for xyname,cellstack in cellstacks_FW.items():
            if cellstack.is_dummy: continue
            dst_path = os.path.join(dst_basedir_FW, "{}_{}.pkl".format(xyname[1],xyname[0]))
            num_cells = process_stack(dst_path, cellstack,
                                      self.wholebrain_images.bound_z_global_FW,
                                      param_scalemerge_FW["merged_margin_left"],
                                      param_scalemerge_FW["merged_margin_top"],
                                      offset_x_FW, offset_y_FW, self.wholebrain_images.new_origin_z_global,
                                      param_scalemerge_FW["actual_downscale_ratio_x"] / org_scale_xy_FW,
                                      param_scalemerge_FW["actual_downscale_ratio_y"] / org_scale_xy_FW,
                                      param_scalemerge_FW["downscale_ratio_z"] / org_scale_z_FW,
                                      A_FW, b_FW)
            dict_num_cells[dst_path] = num_cells

        for xyname,cellstack in cellstacks_RV.items():
            if cellstack.is_dummy: continue
            dst_path = os.path.join(dst_basedir_RV, "{}_{}.pkl".format(xyname[1],xyname[0]))
            num_cells = process_stack(dst_path, cellstack,
                                      self.wholebrain_images.bound_z_global_RV,
                                      param_scalemerge_RV["merged_margin_left"],
                                      param_scalemerge_RV["merged_margin_top"],
                                      offset_x_RV, offset_y_RV, self.wholebrain_images.new_origin_z_global,
                                      param_scalemerge_RV["actual_downscale_ratio_x"] / org_scale_xy_RV,
                                      param_scalemerge_RV["actual_downscale_ratio_y"] / org_scale_xy_RV,
                                      param_scalemerge_RV["downscale_ratio_z"] / org_scale_z_RV,
                                      A_RV, b_RV)
            dict_num_cells[dst_path] = num_cells

        # saving information
        joblib.dump(dict_num_cells, os.path.join(dst_basedir, "info.pkl"), compress=3)
        return

def main():
    args = docopt(__doc__)

    wb_images = WholeBrainImages(args["PARAM_FILE"])

    if args["images"]:
        wb_images.scalemerge(num_cpus=int(args["-p"]), dry_run=False, path_exec=args["--exec"])
    elif args["cells"]:
        wb_images.scalemerge(num_cpus=int(args["-p"]), dry_run=True, path_exec=args["--exec"])
        wb_cells = WholeBrainCells(args["PARAM_FILE"], wholebrain_images=wb_images)
        wb_cells.scalemerge()
    elif args["full"]:
        wb_images.scalemerge(num_cpus=int(args["-p"]), dry_run=False, path_exec=args["--exec"])
        wb_cells = WholeBrainCells(args["PARAM_FILE"], wholebrain_images=wb_images)
        wb_cells.scalemerge()


if __name__ == "__main__":
    main()
