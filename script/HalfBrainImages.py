#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import json, glob, os.path
import functools

class ImageFile(object):
    def __init__(self, stack, fname, is_dummy=False):
        self.stack = stack
        self.fname = fname
        self.fullpath = os.path.join(stack.path, fname)
        self.z = int(os.path.basename(fname).split(".")[0])
        self.is_dummy = is_dummy

    def load_image(self, dtype=np.uint16, height=2160, width=2560):
        if not self.is_dummy:
            img = np.fromfile(self.fullpath, dtype=dtype).reshape(height, width)
        else:
            img = np.zeros((height, width), dtype=dtype)
        return img


class ImageStack(object):
    def __init__(self, path, affine_param, scale_z):
        self.path = path
        self.A_global = np.zeros((4,4))
        self.A_global[:3, :] = np.array(affine_param)
        self.A_global[3, 3] = 1.
        self.scale_z = scale_z
        # assume z is independent from xy
        assert np.count_nonzero(self.A_global[2,(0,1)]) == 0
        assert np.count_nonzero(self.A_global[(0,1),2]) == 0

        self.yxname = os.path.basename(os.path.dirname(os.path.join(path,"")))
        self.yname, self.xname = self.yxname.split("_")

        self.list_fnames_no_dummy = [os.path.basename(fname) for fname in sorted(glob.glob(os.path.join(path, "*.bin")))]
        self.list_zs_no_dummy = [int(os.path.splitext(fname)[0]) for fname in self.list_fnames_no_dummy]
        if len(self.list_fnames_no_dummy) == 0:
            self.list_fnames = []
            self.list_zs = []
            self.offset_x = 0
            self.offset_y = 0
            self.offset_z = 0
            self.list_zs_global_no_dummy = np.zeros((0,))
            self.list_zs_global = np.zeros((0,))
            self.list_imagefiles_no_dummy = []
            self.list_imagefiles = []
            return
        self.list_fnames = [fname for fname in self.list_fnames_no_dummy]
        self.list_zs = [zname for zname in self.list_zs_no_dummy]

        stack_origin = np.array([int(self.xname), int(self.yname), self.list_zs[0], 1]).T
        stack_origin_global = self.A_global.dot(stack_origin)
        self.offset_x = stack_origin_global[0]
        self.offset_y = stack_origin_global[1]
        self.offset_z = stack_origin_global[2]

        if len(self.list_zs_no_dummy) <= 1:
            delta_zs = 1
        else:
            delta_zs = self.list_zs_no_dummy[1] - self.list_zs_no_dummy[0]
        self.list_zs_global_no_dummy = self.offset_z + (np.array(self.list_zs_no_dummy) - self.list_zs_no_dummy[0]) / delta_zs * self.scale_z
        self.list_zs_global = [z for z in self.list_zs_global_no_dummy]
        #print(self.list_zs[0], self.list_zs[-1], delta_zs, self.scale_z, self.offset_z, self.list_zs_global[0], self.list_zs_global[-1])

        self.list_imagefiles_no_dummy = [ImageFile(self, fname) for fname in self.list_fnames]
        # list_imagefiles will be corrected @set_all_fnames()
        self.list_imagefiles = [img for img in self.list_imagefiles_no_dummy]

    def set_all_fnames(self, all_fnames):
        # add dummy image so that all the stacks have the same name set of images
        list_imagefiles = []
        for fname in all_fnames:
            imgfile = self.get_imagefile_by_fname(fname)
            list_imagefiles.append(imgfile)

        self.list_imagefiles = [img for img in list_imagefiles]
        self.list_fnames = all_fnames.copy()
        self.list_zs = [int(os.path.basename(fname).split(".")[0]) for fname in self.list_fnames]
        self.list_zs_global = self.offset_z + (np.array(self.list_zs) - self.list_zs[0]) * self.scale_z
        #print("@set_all_fnames: org:{}, all:{}".format(len(self.list_imagefiles_no_dummy), len(self.list_imagefiles)))
        return

    def get_imagefile_by_fname(self, fname):
        try:
            i_fname = self.list_fnames.index(fname)
            return self.list_imagefiles[i_fname]
        except ValueError:
            return ImageFile(self, fname, is_dummy=True)


class HalfBrainImages(object):
    def __init__(self, paramfile):
        print("[*] Initializing HalfbrainImages({})".format(paramfile))
        with open(paramfile) as f:
            self.params = json.load(f)

        self.scale_x = self.params["coordinate_info"]["scale_x"]
        self.scale_y = self.params["coordinate_info"]["scale_y"]
        self.scale_z = self.params["coordinate_info"]["scale_z"]
        # scale is isometric for xy
        assert abs(self.scale_x) == abs(self.scale_y)
        self.scale_xy = self.scale_x

        # load stack information
        self.dict_imagestacks = {}
        set_xnames = set()
        set_ynames = set()
        set_offset_xs = set()
        set_offset_ys = set()
        set_fnames = set()
        set_zs = set()
        set_zs_global = set()
        for stack_path in glob.glob(os.path.join(self.params["src_basedir"], "*/*")):
            stack = ImageStack(stack_path, self.params["coordinate_info"]["affine_global"], self.scale_z)
            self.dict_imagestacks[(stack.xname,stack.yname)] = stack
            set_xnames.add(stack.xname)
            set_ynames.add(stack.yname)
            set_offset_xs.add(stack.offset_x)
            set_offset_ys.add(stack.offset_y)
            set_fnames.update(stack.list_fnames)
            set_zs.update(stack.list_zs)
            set_zs_global.update(stack.list_zs_global)

        self.list_xnames = sorted(list(set_xnames))
        self.list_ynames = sorted(list(set_ynames))
        self.list_offset_xs = sorted(list(set_offset_xs))
        self.list_offset_ys = sorted(list(set_offset_ys))
        self.list_fnames_all = sorted(list(set_fnames))
        self.list_zs_all = sorted(list(set_zs))
        self.list_zs_global_all = sorted(list(set_zs_global), reverse=self.scale_z < 0)
        #print(self.list_zs_global_all)
        # add dummy image so that all the stacks have the same name set of images
        if len(list_zs_all) > 0:
            print("[*] adding dummy images...")
            for stack in self.dict_imagestacks.values():
                stack.set_all_fnames(self.list_fnames_all)

    def get_imagestack_by_xyname(self, xname, yname):
        #assert xname in self.list_xnames
        #assert yname in self.list_ynames
        stack = self.dict_imagestacks.get((xname,yname), None)
        if not stack:
            raise ValueError
        else:
            return stack
