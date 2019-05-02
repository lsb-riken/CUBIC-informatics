#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Convert tiff file to bin file.

Usage:
  tif2bin.py single <tifffile> <binfile>
  tif2bin.py dirs <tiffdir> <bindir> [-p <num_cpu>]

Options:
  -h --help       Show this screen.
  --version       Show version.
  -p <num_cpu>    Number of cpus to use [default: -1](all available).
"""

from joblib import Parallel, delayed
import numpy as np
import tifffile
import os,os.path,fnmatch,docopt


def tiff2bin_dir(src_dir, dst_dir, _dtype=np.uint16, num_cpus=-1):
    # recursively search for *.tif under {src_dir}
    # then convert and save under {dst_dir}/{the same folder name}/*.bin
    args = []
    for root, dirnames, filenames in os.walk(src_dir):
        for fname in fnmatch.filter(filenames, "*.tif"):
            src_path = os.path.join(root, fname)
            dst_subdir = root.replace(src_dir, dst_dir)
            if not os.path.exists(dst_subdir):
                os.makedirs(dst_subdir)
            dst_path = os.path.join(dst_subdir, fname.replace(".tif", ".bin"))
            args.append((src_path, dst_path))

    Parallel(n_jobs=num_cpus)( [delayed(tiff2bin)(src, dst, _dtype) for src,dst in args] )
    return



def tiff2bin(src_path, dst_path, _dtype=np.uint16):
    print("converting {} to {}".format(src_path, dst_path))
    img = tifffile.imread(src_path)
    np.rot90(img).flatten().tofile(dst_path)
    return

if __name__ == "__main__":
    args = docopt.docopt(__doc__, version="0.1.0")

    if args["single"]:
        tiff2bin(args["<tifffile>"], args["<binfile>"])
    elif args["dirs"]:
        tiff2bin_dir(args["<tiffdir>"], args["<bindir>"], int(args["-p"]))
