#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Save the intermediate results of HDoG as TIFF images

Usage:
  HDoG_intermediate.py PARAM_FILE_MERGE EVALNAME [-o DST_DIR] [--without-normalize] [--without-dog] [--without-hessian]

Options:
  -h --help            Show this screen.
  --version            Show version.
  -o DST_DIR           Set output directory [default: ./](current directory).
  --without-normalize  Do not run Normalize_test.
  --without-dog        Do not run DoG_test.
  --without-hessian    Do not run Hessian_test.
"""

import copy,json,os.path
import subprocess as sp
import numpy as np
import tifffile
from docopt import docopt

from MergeBrain import WholeBrainCells
from evaluation import Evaluation

def Normalize_test(list_src_path, wbc,
                   dst_norm="normalized.bin",
                   dst_erosion="erosion.bin",
                   dst_dilation="dilation.bin", gpuid=0):
    params = copy.deepcopy(wbc.wholebrain_images.halfbrain_FW.params)
    params["devID"] = gpuid
    params["HDoG_param"]["depth"] = 32
    params["stacks"] = [{
        "dst_path":"",
        "src_paths":list_src_path[:32],
    }]
    params["verbose"] = True
    with open("./param_normalizetest.json", "w") as f:
        json.dump(params, f)
    cmd = "test/Normalize_test param_normalizetest.json {} {} {}".format(dst_erosion, dst_dilation, dst_norm)
    print("running: ", cmd)
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    out,err = p.communicate()
    if err:
        print(out)
        print(err)
    else:
        print(out)
    return

def DoG_test(list_src_path, wbc, dst_dog="dog.bin", gpuid=0):
    params = copy.deepcopy(wbc.wholebrain_images.halfbrain_FW.params)
    params["devID"] = gpuid
    params["HDoG_param"]["depth"] = 32
    params["stacks"] = [{
        "dst_path":"",
        "src_paths":list_src_path[:32],
    }]
    params["verbose"] = True
    with open("./param_dogtest.json", "w") as f:
        json.dump(params, f)
    cmd = "test/DoG_test param_dogtest.json {}".format(dst_dog)
    print("running: ", cmd)
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    out,err = p.communicate()
    if err:
        print(out)
        print(err)
    return

def Hessian_test(list_src_path, wbc,
                 dst_dog="dog.bin",
                 dst_pd="hessian_pd.bin",
                 dst_element="hessian.bin", gpuid=0):
    params = copy.deepcopy(wbc.wholebrain_images.halfbrain_FW.params)
    params["devID"] = gpuid
    params["HDoG_param"]["depth"] = 32
    params["stacks"] = [{
        "dst_path":"",
        "src_paths":list_src_path[:32],
    }]
    params["verbose"] = True
    with open("./param_hessiantest.json", "w") as f:
        json.dump(params, f)
    cmd = "test/Hessian_test param_hessiantest.json {} {} {}".format(dst_dog, dst_pd, dst_element)
    print("running: ", cmd)
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    out,err = p.communicate()
    if err:
        print(out)
        print(err)
    return

def save_temporary_results(evalname, wbc, dst_basedir,
                           run_Normalize=True, run_DoG=True, run_Hessian=True
):
    """
    Run each step of HDoG and output those outputs.
    Only support 32 images in a single stack.

    evalname : specify which part of which stack to be used
    e.g. `CB1_on_850-900_148804_249860_1000-1250_750-1000`
    """
    eva = Evaluation(evalname, wbc)
    # select images from top 32 images in substack for evalation
    list_src_path = [eva.imagestack.get_imagefile_by_fname(fname).fullpath
                 for fname in eva.imagestack.list_fnames[eva.local_zlim[0]:eva.local_zlim[0]+32]]
    dst_dir = os.path.join(dst_basedir, eva.region_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if run_Normalize:
        path_norm = os.path.join(dst_dir, "normalized.bin")
        path_ero = os.path.join(dst_dir, "erosion.bin")
        path_dil = os.path.join(dst_dir, "dilation.bin")
    if run_DoG:
        path_dog = os.path.join(dst_dir, "dog.bin")
    if run_Hessian:
        path_pd = os.path.join(dst_dir, "hessian_pd.bin")
        path_element = os.path.join(dst_dir, "hessian.bin")

    # Run
    if run_Normalize:
        Normalize_test(list_src_path, wbc, path_norm, path_ero, path_dil, gpuid=4)
    if run_DoG:
        DoG_test(list_src_path, wbc, path_dog, gpuid=5)
    if run_Hessian:
        Hessian_test(list_src_path, wbc, path_dog, path_pd, path_element, gpuid=6)

    # load binary result
    if run_Normalize:
        erosion_img = np.fromfile(path_ero, dtype=np.float32).reshape(32,2160,2560)[
            :,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]
        dilation_img = np.fromfile(path_dil, dtype=np.float32).reshape(32,2160,2560)[
            :,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]
        norm_img = np.fromfile(path_norm, dtype=np.float32).reshape(32, 2160, 2560)[
            :,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]
    if run_DoG:
        dog_img = np.fromfile(path_dog, dtype=np.float32).reshape(32, 2160, 2560)[
            :,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]
    if run_Hessian:
        pd_img = np.fromfile(path_pd, dtype=np.uint8).reshape(32, 2160, 2560)[
            :,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]
        element_img = np.fromfile(path_element, dtype=np.float32).reshape(6, 32, 2160, 2560)[
            :,:,eva.local_ylim[0]:eva.local_ylim[1],eva.local_xlim[0]:eva.local_xlim[1]]

    # save as tiff
    if run_Normalize:
        tifffile.imsave(
            path_ero.replace(".bin", ".tif"),
            erosion_img
        )
        tifffile.imsave(
            path_dil.replace(".bin", ".tif"),
            dilation_img
        )
        tifffile.imsave(
            path_norm.replace(".bin", ".tif"),
            norm_img
        )
    if run_DoG:
        tifffile.imsave(
            path_dog.replace(".bin", ".tif"),
            dog_img
        )
    if run_Hessian:
        tifffile.imsave(
            path_pd.replace(".bin", ".tif"),
            pd_img
        )
        tifffile.imsave(
            path_element.replace(".bin", "_xx.tif"),
            element_img[0]
        )
        tifffile.imsave(
            path_element.replace(".bin", "_xy.tif"),
            element_img[1]
        )
        tifffile.imsave(
            path_element.replace(".bin", "_xz.tif"),
            element_img[2]
        )
        tifffile.imsave(
            path_element.replace(".bin", "_yy.tif"),
            element_img[3]
        )
        tifffile.imsave(
            path_element.replace(".bin", "_yz.tif"),
            element_img[4]
        )
        tifffile.imsave(
            path_element.replace(".bin", "_zz.tif"),
            element_img[5]
        )
    return

if __name__ == "__main__":
    args = docopt(__doc__)

    wbc = WholeBrainCells(args["PARAM_FILE_MERGE"])
    save_temporary_results(args["EVALNAME"], wbc,
                           args["-o"],
                           run_Normalize=(not args["--without-normalize"]),
                           run_DoG=(not args["--without-normalize"]),
                           run_Hessian=(not args["--without-normalize"]))
