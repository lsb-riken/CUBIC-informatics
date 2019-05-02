#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Convert bin file to tiff file.

Usage:
  bin2tif.py single <binfile> <tifffile>
  bin2tif.py dirs <bindir> <tiffdir> [-p <num_cpu>]
  bin2tif.py stack <bindir> <tifffile> [--first <first>] [--last <last>] [--skip <skip>]

Options:
  -h --help        Show this screen.
  --version        Show version.
  --first <first>  Index of the first in the stack [default: 0].
  --last <last>    Index of the last in the stack [default: -1](last found).
  --skip <skip>    Skip in the stack [default: 1](no skip).
  -p <num_cpu>     Number of cpus to use [default: -1](all available).
"""

from joblib import Parallel, delayed
import numpy as np
import tifffile
import os,os.path,fnmatch, docopt


def bin2tiff(src_path, dst_path, _dtype=np.uint16):
    print("converting {} to {}".format(src_path, dst_path))
    img = np.fromfile(src_path, dtype=_dtype)
    img = np.rot90(img.reshape(2160, 2560), k=-1)
    tifffile.imsave(dst_path, img, compress=0, photometric="minisblack", planarconfig="contig", resolution=(72,72))
    return

def bin2tiff_multipage(src_dir, dst_path, start=0, end=-1, skip=1, _dtype=np.uint16):
    # serch for {src_dir}/*.bin  (1 stack)
    # then convert and save to one multipage tiff ({dst_path})
    src_fnames = sorted(os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if fname.endswith(".bin"))
    if end < 0:
        end = len(src_fnames)
    src_fnames = src_fnames[start:end:skip]
    N = len(src_fnames)
    imgs = np.empty((N, 2560, 2160), dtype=_dtype)
    for i,fname in enumerate(src_fnames):
        img = np.fromfile(fname, dtype=_dtype)
        img = np.rot90(img.reshape(2160, 2560), k=-1)
        imgs[i,:,:] = img
        print(i, fname)
    tifffile.imsave(dst_path, imgs)
    return

def bin2tiff_dir(src_dir, dst_dir, _dtype=np.uint16, num_cpus=-1):
    # recursively search for *.bin under {src_dir}
    # then convert and save under {dst_dir}/{the same folder name}/*.tif
    args = []
    for root,dirnames,filenames in os.walk(src_dir):
        for fname in fnmatch.filter(filenames, "*.bin"):
            src_path = os.path.join(root, fname)
            dst_subdir = root.replace(src_dir, dst_dir)
            if not os.path.exists(dst_subdir):
                os.makedirs(dst_subdir)
            dst_path = os.path.join(dst_subdir, fname.replace(".bin", ".tif"))
            args.append((src_path, dst_path))
    Parallel(n_jobs=num_cpus)( [delayed(bin2tiff)(src,dst,_dtype=_dtype) for src,dst in args] )
    return

if __name__ == "__main__":
    args = docopt.docopt(__doc__, version="0.1.0")

    if args["single"]:
        bin2tiff(args["<binfile>"], args["<tifffile>"])
    elif args["dirs"]:
        bin2tiff_dir(args["<bindir>"], args["<tiffdir>"], num_cpus=int(args["-p"]))
    elif args["stack"]:
        bin2tiff_multipage(args["<bindir>"], args["<tifffile>"], int(args["--first"]), int(args["--last"]), int(args["--skip"]))
