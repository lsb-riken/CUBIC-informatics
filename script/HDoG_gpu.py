#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Assign images and launch `HDoG3D_main` in parallel

Usage:
  HDoG_gpu PARAM_FILE [--zmin <ZMIN>] [--zmax <ZMAX>] [--exec <path>]
  HDoG_gpu PARAM_FILE single <xname> <yname> [--zmin <ZMIN>] [--zmax <ZMAX>] [--exec <path>]

Options:
  PARAM_FILE     : Parameter file
  --zmin <ZMIN>  : Skip first <ZMIN> images from top (optional)
  --zmax <ZMAX>  : Skip after <ZMAX> images from top (optional)
  --exec <path>  : Specify the location of the executable (optional) [default: /usr/local/bin/HDoG3D_main]
"""

import os, os.path, glob, sys
import tempfile, json, copy
import numpy as np
import subprocess as sp
import multiprocessing as mp
from docopt import docopt


def run(args):
    paramfile, logfile, path_HDoG_executable = args
    cmd = "{} {}".format(path_HDoG_executable, paramfile)
    print("Executing : {}".format(cmd))
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
    out,_ = p.communicate()
    with open(logfile, "wb") as f:
        f.write(out)
    return

def main():
    args = docopt(__doc__)
    try:
        min_z = int(args["--zmin"])
    except (ValueError,TypeError):
        min_z = None
    try:
        max_z = int(args["--zmax"])
    except (ValueError,TypeError):
        max_z = None

    with open(args["PARAM_FILE"]) as f:
        params = json.load(f)

    num_gpu = len(params["gpus"])
    result_dir = os.path.join(params["dst_basedir"])
    try:
        os.makedirs(result_dir)
    except FileExistsError:
        pass

    # src image path format : {src_basedir}/{Yname}/{Yname}_{Xname}/{Zname}.bin
    # e.g. {src_basedir}/095000/095000_150000/201359.bin
    if args["single"]:
        stack_path = os.path.join(params["src_basedir"], "{yname}/{yname}_{xname}".format(xname=args["<xname>"], yname=args["<yname>"]))
        list_stack = [(stack_path, len(glob.glob(os.path.join(stack_path, "*.bin"))[min_z:max_z]))]
    else:
        list_stack = [
            (stack_path, len(glob.glob(os.path.join(stack_path,"*.bin"))[min_z:max_z]))
            for stack_path in glob.glob(os.path.join(params["src_basedir"], "*/*"))
        ]
    list_stack.sort(key=lambda x:x[1], reverse=True)
    num_stacks = len(list_stack)
    dict_stack_per_gpu = {}
    dict_num_images_per_gpu = {}
    for i,gpu_id in enumerate(params["gpus"]):
        dict_stack_per_gpu[gpu_id] = []
        dict_num_images_per_gpu[gpu_id] = 0
        for i_stack in range(i, num_stacks, num_gpu):
            dict_stack_per_gpu[gpu_id].append(list_stack[i_stack][0])
            dict_num_images_per_gpu[gpu_id] += list_stack[i_stack][1]
        print("GPU:{}\timages:{}".format(gpu_id,dict_num_images_per_gpu[gpu_id]))

    list_paramfile_gpus = []
    list_logfile_gpus = []
    for gpu_id in params["gpus"]:
        paramfile_gpu = os.path.join(os.path.dirname(args["PARAM_FILE"]), "param_gpu{:02d}.json".format(gpu_id))
        logfile_gpu = paramfile_gpu.replace("param_", "log_").replace(".json", ".txt")
        list_paramfile_gpus.append(paramfile_gpu)
        list_logfile_gpus.append(logfile_gpu)
        print("GPU:{}\tparamfile: {}".format(gpu_id,paramfile_gpu))
        with open(paramfile_gpu, "w") as f:
            params_gpu = copy.deepcopy(params)
            params_gpu["devID"] = gpu_id
            params_gpu["stacks"] = []
            for stack_path in dict_stack_per_gpu[gpu_id]:
                yxname = os.path.basename(os.path.dirname(os.path.join(stack_path,"")))
                znames = sorted(glob.glob(os.path.join(stack_path, "*.bin")))[min_z:max_z]
                stack_param = { # in [um]
                    "dst_path":os.path.join(result_dir,"{}.bin".format(yxname)),
                    "src_paths":[os.path.join(stack_path, zname) for zname in znames]
                }
                params_gpu["stacks"].append(stack_param)
            json.dump(params_gpu, f, indent=4)

    list_path_executables = [args["--exec"]]*len(params["gpus"])
    pool = mp.Pool(processes=num_gpu)
    for out in pool.map(run, zip(list_paramfile_gpus, list_logfile_gpus, list_path_executables)):
        print("[*] finished")
        #print(out)

    # remove temporary files
    for fname in list_paramfile_gpus:
        os.remove(fname)

if __name__ == "__main__":
    main()
