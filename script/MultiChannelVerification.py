#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Multi-channel verification by intensity ratio.

Usage:
  MultiChannelVerification.py PARAM_FILE [-p NUM_CPUS]

Options:
  -h --help      Show this screen.
  --version      Show version.
  -p NUM_CPUS    Number of CPUs to be used. [default: 10]
"""

import numpy as np
import pandas as pd
import joblib
import os.path
from docopt import docopt
from MergeBrain import WholeBrainCells
from evaluation import *
from HDoG_classifier import *
from HalfBrainCells import *

def save_intensities(
        xyname,
        resultfile_nucl,
        imagestack_nucl, imagestack_second,
        clf, dst_basedir):
    data_local_nucl = np.fromfile(resultfile_nucl, dtype=dt_local)
    if data_local_nucl.shape[0] == 0: return
    _X = get_X_3d(data_local_nucl)
    #print(_X.shape)
    _pred = clf.predict(_X)
    centroids = np.floor(np.array([
        data_local_nucl[_pred]["local_z"],
        data_local_nucl[_pred]["local_y"],
        data_local_nucl[_pred]["local_x"]
    ]).T).astype(int)
    #print(centroids.shape)

    src_img_nucl = np.zeros((len(imagestack_nucl.list_imagefiles_no_dummy),
                             2160,2560), dtype=np.uint16)
    src_img_second = np.zeros((len(imagestack_second.list_imagefiles_no_dummy),
                             2160,2560), dtype=np.uint16)
    for z,imgfile in enumerate(imagestack_nucl.list_imagefiles_no_dummy):
        src_img_nucl[z,:,:] = imgfile.load_image()
    for z,imgfile in enumerate(imagestack_second.list_imagefiles_no_dummy):
        src_img_second[z,:,:] = imgfile.load_image()[::-1,:]

    intensities_nucl = src_img_nucl[centroids[:,0],centroids[:,1],centroids[:,2]]
    intensities_second = src_img_second[centroids[:,0],centroids[:,1],centroids[:,2]]
    intensities = np.array([
        np.log10(intensities_nucl),
        np.log10(intensities_second)
    ]).T

    intensities_full = np.zeros((data_local_nucl.shape[0],2), dtype=np.float32)
    intensities_full[:,:] = np.nan
    intensities_full[_pred,:] = intensities

    joblib.dump(intensities_full, os.path.join(dst_basedir, "_".join(xyname)+".pkl"))


def main():
    args = docopt(__doc__)

    with open(args["PARAM_FILE"]) as f:
        params_multichannel = json.load(f)
    wbc = WholeBrainCells(params_multichannel["paramfile_nucl"])
    wbc2 = WholeBrainCells(params_multichannel["paramfile_second"])

    clf_manual3 = joblib.load(params_multichannel["clf_file"])
    result_dir_FW = os.path.join(wbc.halfbrain_cells_FW.halfbrain_images.params["dst_basedir"])
    result_dir_RV = os.path.join(wbc.halfbrain_cells_RV.halfbrain_images.params["dst_basedir"])
    if not os.path.exists(os.path.join(params_multichannel["dst_basedir"], "FW")):
        os.makedirs(os.path.join(params_multichannel["dst_basedir"], "FW"))
    if not os.path.exists(os.path.join(params_multichannel["dst_basedir"], "RV")):
        os.makedirs(os.path.join(params_multichannel["dst_basedir"], "RV"))

    joblib.Parallel(n_jobs=args["-p"], verbose=10)( [
        joblib.delayed(save_intensities)(
            xyname = (xname,yname),
            resultfile_nucl = os.path.join(result_dir_FW, "{}_{}.bin".format(yname,xname)),
            imagestack_nucl = wbc.halfbrain_cells_FW.dict_stacks[(xname,yname)].imagestack,
            imagestack_second = wbc2.halfbrain_cells_FW.dict_stacks[(xname,yname)].imagestack,
            clf = clf_manual3,
            dst_basedir = os.path.join(params_multichannel["dst_basedir"], "FW")
        )
        for (xname,yname) in wbc.halfbrain_cells_FW.dict_stacks.keys()
        if os.path.exists(os.path.join(result_dir_FW, "{}_{}.bin".format(yname,xname)))
    ] + [
        joblib.delayed(save_intensities)(
            xyname = (xname,yname),
            resultfile_nucl = os.path.join(result_dir_RV, "{}_{}.bin".format(yname,xname)),
            imagestack_nucl = wbc.halfbrain_cells_RV.dict_stacks[(xname,yname)].imagestack,
            imagestack_second = wbc2.halfbrain_cells_RV.dict_stacks[(xname,yname)].imagestack,
            clf = clf_manual3,
            dst_basedir = os.path.join(params_multichannel["dst_basedir"], "RV")
        )
        for (xname,yname) in wbc.halfbrain_cells_RV.dict_stacks.keys()
        if os.path.exists(os.path.join(result_dir_RV, "{}_{}.bin".format(yname,xname)))
    ])

if __name__ == "__main__":
    main()
