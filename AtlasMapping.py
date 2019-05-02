#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Overview:
  Registration and mapping cells onto CUBIC-Atlas

Usage:
  AtlasMapping.py registration PARAM_FILE [-p NUM_CPUS]
  AtlasMapping.py annotation PARAM_FILE [-p NUM_CPUS]
  AtlasMapping.py full PARAM_FILE [-p NUM_CPUS]

Options:
  -h --help      Show this screen.
  --version      Show version.
  -p NUM_CPUS    Number of CPUs used for ANTs. [default: 20]
"""

import json, os.path, os, re, time
import tifffile
import joblib
from docopt import docopt
import subprocess as sp
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.spatial

from HalfBrainCells import dt_local
from HDoG_classifier import get_X_3d

dt_annotated = np.dtype([
    ('mapped_x', 'f4'), ('mapped_y', 'f4'), ('mapped_z', 'f4'),
    ('atlas_id', 'u2')
])

def run_antsRegistration(prefix_ants, atlas_file, moving_file, dst_dir, threads):

    cmd = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={THREADS} && "
    cmd += "{EXECUTABLE} -d 3 "
    cmd += "--initial-moving-transform [{ATLAS_FILE},{MOVING_FILE},1] "
    cmd += "--interpolation Linear "
    cmd += "--use-histogram-matching 0 "
    cmd += "--float 0 "
    cmd += "--output [{DST_PREFIX},{WARPED_FILE},{INVWARPED_FILE}] "
    cmd += "--transform Affine[0.1] --metric MI[{ATLAS_FILE},{MOVING_FILE},1,128,Regular,0.5] --convergence [10000x10000x10000,1e-5,15] --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox "
    cmd += "--transform SyN[0.1,3.0,0.0] --metric CC[{ATLAS_FILE},{MOVING_FILE},1,5] --convergence [300x100x30,1e-6,10] --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox"
    cmd = cmd.format(
        THREADS = threads,
        EXECUTABLE = os.path.join(prefix_ants, "antsRegistration"),
        DST_PREFIX = os.path.join(dst_dir, "F2M_"),
        WARPED_FILE = os.path.join(dst_dir, "F2M_Warped.nii.gz"),
        INVWARPED_FILE = os.path.join(dst_dir, "F2M_InvWarped.nii.gz"),
        ATLAS_FILE = atlas_file,
        MOVING_FILE = moving_file,
    )
    print("[*] Executing : {}".format(cmd))
    sp.call(cmd, shell=True)
    return

def run_antsApplyTransformsToPoints(prefix_ants, src_csv, dst_csv, dst_dir):
    cmd = "{EXECUTABLE} "
    cmd += "-d 3 "
    cmd += "-i {SRC_CSV} "
    cmd += "-o {DST_CSV} "
    cmd += "-t [{AFFINE_MAT},1] "
    cmd += "-t {INVWARP_NII}"
    cmd = cmd.format(
        EXECUTABLE = os.path.join(prefix_ants, "antsApplyTransformsToPoints"),
        AFFINE_MAT = os.path.join(dst_dir, "F2M_0GenericAffine.mat"),
        INVWARP_NII = os.path.join(dst_dir, "F2M_1InverseWarp.nii.gz"),
        SRC_CSV = src_csv,
        DST_CSV = dst_csv,
    )
    #print("[*] Executing : {}".format(cmd))
    # supress output
    with open(os.devnull, 'w') as devnull:
        sp.check_call(cmd, shell=True, stdout=devnull)
    return

def register(atlas_basedir, merging_basedir, mapping_basedir,
             prefix_ANTs, atlas_voxel_unit, num_cpus=36,
             atlas_basename="iso_80um",
):
    atlas_tif_path = os.path.join(atlas_basedir, "{}.tif".format(atlas_basename))
    atlas_nii_path = os.path.join(atlas_basedir, "{}.nii.gz".format(atlas_basename))
    moving_tif_path = os.path.join(merging_basedir, "whole.tif")
    moving_nii_path = os.path.join(mapping_basedir, "whole.nii.gz")

    # prepare nifti image for atlas
    print("[*] Preparing nifti image for atlas...")
    img_atlas = tifffile.imread(atlas_tif_path)
    if not os.path.exists(atlas_nii_path):
        nii_atlas = nib.Nifti1Image(np.swapaxes(img_atlas,0,2), affine=None)
        aff = np.diag([-atlas_voxel_unit,-atlas_voxel_unit,atlas_voxel_unit,1])
        nii_atlas.header.set_qform(aff, code=2)
        nii_atlas.to_filename(atlas_nii_path)
    # prepare nifti image for moving
    print("[*] Preparing nifti image for moving...")
    img_moving = tifffile.imread(moving_tif_path)
    nii_moving = nib.Nifti1Image(np.swapaxes(img_moving,0,2), affine=None)
    aff = np.diag([-atlas_voxel_unit,-atlas_voxel_unit,atlas_voxel_unit,1])
    nii_moving.header.set_qform(aff, code=2)
    nii_moving.to_filename(moving_nii_path)

    # run registration
    run_antsRegistration(prefix_ants = prefix_ANTs,
                         atlas_file = atlas_nii_path,
                         moving_file = moving_nii_path,
                         dst_dir = mapping_basedir,
                         threads = num_cpus)
    return

def map_and_annotate_cellstacks(list_src_pkl_path, list_annotated_pkl_path, total_num_cells,
                                prefix_ANTs, mapping_basedir, atlas_points_path,
                                downscale_unit, HDoG_basedir, clf, max_distance):
    # apply transforms and annotate cells in stacks

    # initialize
    print("[{}] Loading point atlas and constructing KD Tree...".format(os.getpid()))
    if atlas_points_path.endswith("pkl"):
        df_atlas = joblib.load(atlas_points_path)
    elif atlas_points_path.endswith("csv") or atlas_points_path.endswith("csv.gz"):
        df_atlas = pd.read_csv(atlas_points_path, skiprows=1, header=None,
                               names=["X(um)","Y(um)","Z(um)","atlasID"],
                               dtype={"X(um)":np.float32, "Y(um)":np.float32, "Z(um)":np.float32, "atlasID":np.uint16})

    tree = scipy.spatial.cKDTree( np.array([
        df_atlas["X(um)"].values,
        df_atlas["Y(um)"].values,
        df_atlas["Z(um)"].values
    ]).T )
    print("[{}] KD Tree Construction completed.".format(os.getpid()))
    pat = re.compile(os.path.join(r'(?P<FWRV>FW|RV)', r'(?P<XYNAME>\d+_\d+)\.pkl$'))
    count = 0
    for src_pkl_path,annotated_pkl_path in zip(list_src_pkl_path, list_annotated_pkl_path):
        start = time.time()
        print("[{}]({:.2f}%| {:.0f}s) Loading scalemerged data ({})...".format(
            os.getpid(), float(count)/total_num_cells*100,
            time.time()-start, src_pkl_path))
        data_scalemerged = joblib.load(src_pkl_path)
        if data_scalemerged.shape[0] == 0:
            print("[{}]({:.2f}%| {:.0f}s) No data points. skipping".format(
                os.getpid(), float(count)/total_num_cells*100), time.time()-start)
            continue

        # use predicted cells if classifier is specified
        if clf is not None:
            m = pat.search(src_pkl_path)
            if not m: raise ValueError
            HDoG_bin_path = os.path.join(HDoG_basedir[m.group("FWRV")], m.group("XYNAME")+".bin")
            print("[{}]({:.2f}%| {:.0f}s) Loading HDoG local data ({})...".format(
                os.getpid(), float(count)/total_num_cells*100,
                time.time()-start, HDoG_bin_path))
            data_local = np.fromfile(HDoG_bin_path, dtype=dt_local)
            X = get_X_3d(data_local)
            pred = clf.predict(X)
            is_valid = np.bitwise_and(pred, data_scalemerged["is_valid"])
        else:
            is_valid = data_scalemerged["is_valid"]
        print("[{}]({:.2f}%| {:.0f}s) {:.1f} % valid data points.".format(
            os.getpid(), float(count)/total_num_cells*100, time.time()-start,
            float(np.count_nonzero(is_valid))/is_valid.shape[0]*100))
        data_scalemerged_valid = data_scalemerged[is_valid]
        if data_scalemerged_valid.shape[0] == 0:
            print("[{}]({:.2f}%| {:.0f}s) No valid data points. skipping".format(
                os.getpid(), float(count)/total_num_cells*100, time.time()-start))
            continue

        # write out coordinates as csv file for transformation
        print("[{}]({:.2f}%| {:.0f}s) Transforming points...".format(
            os.getpid(), float(count)/total_num_cells*100,
            time.time()-start))
        df = pd.DataFrame({
            "X(um)":pd.Series(data_scalemerged_valid["scaled_x"]*downscale_unit, dtype=np.float32),
            "Y(um)":pd.Series(data_scalemerged_valid["scaled_y"]*downscale_unit, dtype=np.float32),
            "Z(um)":pd.Series(data_scalemerged_valid["scaled_z"]*downscale_unit, dtype=np.float32)
        })
        FWRV = os.path.basename(os.path.dirname(src_pkl_path))
        basename = os.path.basename(src_pkl_path).replace(".pkl", ".csv")
        tmp_csv_path = "/tmp/AtlasMapping-moving-{}-{}".format(FWRV, basename)
        df.to_csv(tmp_csv_path, index=False, header=True, chunksize=50000,
                  columns=["X(um)","Y(um)","Z(um)"], float_format="%.3f")

        transformed_csv_path = annotated_pkl_path.replace(".pkl", ".csv")
        run_antsApplyTransformsToPoints(
            prefix_ants = prefix_ANTs,
            src_csv = tmp_csv_path,
            dst_csv = transformed_csv_path,
            dst_dir = mapping_basedir)

        os.remove(tmp_csv_path)

        print("[{}]({:.2f}%| {:.0f}s) Loading transformed csv({})...".format(
            os.getpid(), float(count)/total_num_cells*100,
            time.time()-start, transformed_csv_path))
        df_transformed = pd.read_csv(
            transformed_csv_path,
            dtype={"X(um)":np.float32, "Y(um)":np.float32, "Z(um)":np.float32}
        )
        # start annotating
        print("[{}]({:.2f}%| {:.0f}s) Starting annotation...".format(
            os.getpid(), float(count)/total_num_cells*100, time.time()-start))
        dist, idx = tree.query( np.array([
            df_transformed["X(um)"].values,
            df_transformed["Y(um)"].values,
            df_transformed["Z(um)"].values,
        ]).T, k=1, eps=0, p=2, distance_upper_bound=max_distance)

        os.remove(transformed_csv_path)
        print("[{}]({:.2f}%| {:.0f}s) Finished annotation...".format(
            os.getpid(), float(count)/total_num_cells*100, time.time()-start))

        # save result
        print("[{}]({:.2f}%| {:.0f}s) Saving annotated result to {}...".format(
            os.getpid(), float(count)/total_num_cells*100,
            time.time()-start, annotated_pkl_path))
        print("[{}]({:.2f}%| {:.0f}s) There are {} orphan points.".format(
            os.getpid(), float(count)/total_num_cells*100, time.time()-start,
            np.count_nonzero(idx == tree.n)))
        atlas_ID = np.zeros(idx.shape)
        print("df_atlas.shape:", df_atlas["atlasID"].values.shape,
              "idx.shape:",idx.shape, "idx[idx!=tree.n].shape:", idx[idx != tree.n].shape)
        atlas_ID[idx != tree.n] = df_atlas["atlasID"].values[idx[idx != tree.n]]
        data_annotated = np.empty(data_scalemerged.shape[0], dtype=dt_annotated)
        data_annotated["mapped_x"] = np.nan
        data_annotated["mapped_y"] = np.nan
        data_annotated["mapped_z"] = np.nan
        data_annotated["atlas_id"] = 0
        data_annotated["mapped_x"][is_valid] = df_transformed["X(um)"]
        data_annotated["mapped_y"][is_valid] = df_transformed["Y(um)"]
        data_annotated["mapped_z"][is_valid] = df_transformed["Z(um)"]
        data_annotated["atlas_id"][is_valid] = atlas_ID

        joblib.dump(data_annotated, annotated_pkl_path, compress=3)

        count += np.count_nonzero(data_scalemerged["is_valid"])

    print("[{}]({:.2f}%) Finished all the jobs!".format(os.getpid(), float(count)/total_num_cells*100))
    return


def main():
    args = docopt(__doc__)

    with open(args["PARAM_FILE"]) as f:
        params_mapping = json.load(f)
    with open(params_mapping["MergeBrain_paramfile"]) as f:
        params_merge = json.load(f)
    assert params_merge["scale_info"]["downscale_unit"] == params_mapping["atlas_voxel_unit"]
    atlas_voxel_unit = params_mapping["atlas_voxel_unit"]
    with open(params_merge["HDoG_paramfile"]["FW"]) as f:
        params_HDoG_FW = json.load(f)
    with open(params_merge["HDoG_paramfile"]["RV"]) as f:
        params_HDoG_RV = json.load(f)
    HDoG_basedir = {
        "FW":params_HDoG_FW["dst_basedir"],
        "RV":params_HDoG_RV["dst_basedir"],
    }
    atlas_basedir = params_mapping["atlas_folder"]
    merging_basedir = params_merge["dst_basedir"]
    mapping_basedir = params_mapping["dst_basedir"]
    num_cpus = int(args["-p"])
    mapping_basedir_FW = os.path.join(mapping_basedir, "FW")
    mapping_basedir_RV = os.path.join(mapping_basedir, "RV")
    if not os.path.exists(mapping_basedir_FW):
        os.makedirs(mapping_basedir_FW)
    if not os.path.exists(mapping_basedir_RV):
        os.makedirs(mapping_basedir_RV)

    if args["registration"] or args["full"]:
        # -----
        # Image-based registration
        # -----
        register(atlas_basedir,
                 merging_basedir,
                 mapping_basedir,
                 params_mapping["prefix_ANTs"],
                 atlas_voxel_unit,
                 num_cpus,
                 atlas_basename=params_mapping["atlas_img_basename"],
        )

    if args["annotation"] or args["full"]:
        # ------
        # Annotation to the Point-based Atlas
        # ------

        # load classifier if specified
        clf_path = params_mapping["clf_file"]
        if os.path.exists(clf_path):
            print("[*] classifier is specified({})".format(clf_path))
            clf = joblib.load(clf_path)
        else:
            print("[!] classfier is not specified.")
            clf = None

        atlas_points_path = os.path.join(atlas_basedir, "{}.pkl".format(params_mapping["atlas_points_basename"]))
        if not os.path.exists(atlas_points_path):
            atlas_points_path = atlas_points_path.replace(".pkl", ".csv.gz")
            if not os.path.exists(atlas_points_path):
                atlas_points_path = atlas_points_path.replace(".csv.gz", ".csv")
                if not os.path.exists(atlas_points_path):
                    raise FileNotFoundError

        dict_num_cells = joblib.load(os.path.join(merging_basedir, "info.pkl"))
        # Assign cells to each job
        # By sorting cellstacks by number of cells,
        # every job has roughly equal number of assigned cells
        joblist_moving_pkl_path = [[] for i in range(num_cpus)]
        joblist_annotated_pkl_path = [[] for i in range(num_cpus)]
        job_num_cells = [0 for i in range(num_cpus)]
        for i,(moving_pkl_path, num_cells) in enumerate(sorted(dict_num_cells.items(), key=lambda x:x[1], reverse=True)):
            if int(i / num_cpus) % 2 == 0:
                jobid = i % num_cpus
            else:
                jobid = (-i-1) % num_cpus


            joblist_moving_pkl_path[jobid].append(moving_pkl_path)
            parent_dirname = os.path.basename(os.path.dirname(moving_pkl_path))
            if  parent_dirname == "FW":
                mapping_basedir_FWRV = mapping_basedir_FW
            elif parent_dirname == "RV":
                mapping_basedir_FWRV = mapping_basedir_RV
            else:
                raise ValueError
            annotated_pkl_path = os.path.join(mapping_basedir_FWRV, os.path.basename(moving_pkl_path))
            joblist_annotated_pkl_path[jobid].append(annotated_pkl_path)
            #print(i,num_cells,moving_pkl_path,annotated_pkl_path)
            job_num_cells[jobid] += num_cells

        for jobid in range(num_cpus):
            print("job{}:\t{:} cells".format(jobid, job_num_cells[jobid]))

        joblib.Parallel(n_jobs=num_cpus)( [
            joblib.delayed(map_and_annotate_cellstacks)(
                list_src_pkl_path = job_moving_pkl_path,
                list_annotated_pkl_path = job_annotated_pkl_path,
                total_num_cells = num_cells,
                prefix_ANTs = params_mapping["prefix_ANTs"],
                mapping_basedir = mapping_basedir,
                atlas_points_path = atlas_points_path,
                downscale_unit = atlas_voxel_unit,
                HDoG_basedir = HDoG_basedir,
                clf = clf,
                max_distance = params_mapping["max_distance"],
            )
            for job_moving_pkl_path,job_annotated_pkl_path,num_cells in zip(joblist_moving_pkl_path,joblist_annotated_pkl_path,job_num_cells)])



if __name__ == "__main__":
    main()
