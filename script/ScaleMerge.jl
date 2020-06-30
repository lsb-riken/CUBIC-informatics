doc = """Overview:
  Downscale & merge input images to tiff files.

Usage:
  ScaleMerge.jl <srcdir> <outdir> [--tif|--bin]
                [--width <w>] [--height <h>] [--overlap <L:R:T:B>]
                [--scale <ratio>] [--skip-z <num>] [--grid]
                [--CW-rot|--CCW-rot] [--V-flip] [--H-flip] [--flip-V] [--flip-H]
                [--config <json>]
  ScaleMerge.jl (-h | --help)
  ScaleMerge.jl --version

Options:
  -h, --help             Show this screen.
  --version              Show version.
  --tif                  Search for input images in .tif format.
  --bin                  Search for input images in .bin format.
  --width <w>            Width of the rotated image[default: 2160].
  --height <h>           Height of the rotated image[default: 2560].
  --overlap <L:R:T:B>    Overlaping pixels[default: 100:100:100:100].
  --scale <ratio>        Downscale by this ratio[default: 0.05].
  --skip-z <num>         Skip images along z axis[default: 1].
  --grid                 Overlay grid with input image size.
  --CW-rot               Rotate by 90 degrees clockwise before merging.
  --CCW-rot              Rotate by 90 degrees counter clockwise before merging.
  --V-flip               Flip vertically before merging.
  --H-flip               Flip horizontally before merging.
  --flip-V               Flip vertically after merging.
  --flip-H               Flip horizontally after merging.
  --config <json>        Load configuration in json format.
"""

using DocOpt
using Nullables
import JSON

function searchImages(basedir::String, imgformat::Int)
    yset = Set{String}()
    xset = Set{String}()
    zset = Set{String}()
    imgdict = Dict{String,Dict}()
    ext = (imgformat == 0) ? ".tif" : ".bin"
    for yname in sort(readdir(basedir))
        ybasedir = joinpath(basedir, yname)
        if !isdir(ybasedir)
            continue
        end
        push!(yset, yname)
        imgdict[yname] = Dict{String,Array}()
        for yxname in sort(readdir(ybasedir))
            xybasedir = joinpath(ybasedir, yxname)
            if !isdir(xybasedir)
                continue
            end
            xname = last(split(yxname, '_'))
            push!(xset, xname)
            imgdict[yname][xname] = sort(map(fname->fname[1:end-4],
                                             filter(fname->endswith(fname, ext),
                                                    readdir(xybasedir)
                                                    )
                                             )
                                         )
            union!(zset, imgdict[yname][xname])
        end
    end
    ylist = sort(collect(yset))
    xlist = sort(collect(xset))
    zlist = sort(collect(zset))
    imgdict,ylist,xlist,zlist
end

function writeParamFile(param_path::String, z::String, imgdict::Dict, ylist::Array, xlist::Array, basedir::String, width::Int, height::Int, sampling_rate::Float32, overlap::String, flip_rot_before::Int, flip_rot_after::Int, imgformat::Int, showgrid::Int)
    len_x = length(xlist)
    len_y = length(ylist)
    open(param_path, "w") do f
        write(f, "$(width):$(height):$(len_x):$(len_y):$(sampling_rate):$(overlap):$(flip_rot_before):$(flip_rot_after):$(imgformat):$(showgrid):\n")
        #println("imgdict keys:", keys(imgdict))
        for y in ylist
            #println("imgdict[$y] keys:", keys(imgdict[y]))
            for x in xlist
                #println("imgdict[$y][$x] array", imgdict[y][x])
                if z in imgdict[y][x]
                    #println((basedir, imgdict[y][x], y, "$(y)_$(x)", z))
                    imgpath = joinpath(basedir, y, "$(y)_$(x)", string(z, (imgformat==0) ? ".tif" : ".bin"))
                    write(f, "$(imgpath)\n")
                else
                    write(f, "\n")
                end
            end
        end
    end
end

function main()
    arguments = docopt(doc, version=v"0.2.0")
    println("arguments:")
    #for (k,v) in arguments
    #    println("\t$(k) : $(v)")
    #end

    if arguments["--config"] != nothing
        j = JSON.parsefile(arguments["--config"])
        #println("config file:")
        #for (k,v) in j
        #    println("\t$(k) : $(v)")
        #end
        arguments = merge(j, arguments)
        #println("arguments after merged:")
        #for (k,v) in arguments
        #    println("\t$(k) : $(v)")
        #end
    end
    basedir = arguments["<srcdir>"]
    outdir = arguments["<outdir>"]
    width = parse(Int, arguments["--width"])
    height = parse(Int, arguments["--height"])
    overlap = arguments["--overlap"]
    if any(map(x->tryparse(Float64,x) === nothing, split(overlap, ":")))
        println("invalid overlap syntax.")
    end
    skip_z = parse(Int, arguments["--skip-z"])
    sampling_rate = parse(Float32, arguments["--scale"])
    flip_rot_before = 0
    flip_rot_before += (arguments["--H-flip"] ? 1 : 0)
    flip_rot_before += (arguments["--V-flip"] ? 2 : 0)
    flip_rot_before += (arguments["--CCW-rot"] ? 4 : 0)
    flip_rot_before += (arguments["--CW-rot"] ? 8 : 0)
    flip_rot_after = 0
    flip_rot_after += (arguments["--flip-H"] ? 1 : 0)
    flip_rot_after += (arguments["--flip-V"] ? 2 : 0)
    imgformat = arguments["--tif"] ? 0 : 1
    showgrid = arguments["--grid"] ? 1 : 0

    println("basedir:", basedir)
    println("outdir:", outdir)
    imgdict,ylist,xlist,zlist = searchImages(basedir, imgformat)
    if !isdir(outdir)
        mkpath(outdir)
    end

    len_z = length(zlist)
    len_z_skip = Int(round(len_z / skip_z))
    println("max z length: $len_z")
    println("skipped z length: $len_z_skip")

    randomID = rand(UInt32)
    zlist_skip = zlist[1:skip_z:len_z]
    paramfiles = ["/tmp/param_ScaleMerge_$(randomID)_$zname.txt" for zname in zlist_skip]
    mergedfiles = [joinpath(outdir, "$zname.tif") for zname in zlist_skip]

    for (zname,paramfile) in zip(zlist_skip, paramfiles)
        writeParamFile(paramfile, zname, imgdict, ylist, xlist, basedir,
                       width, height, sampling_rate, String(overlap), flip_rot_before, flip_rot_after, imgformat, showgrid)
    end
    pmap((paramfile,mergedfile)->run(`./ScaleMerge $paramfile $mergedfile`), paramfiles,mergedfiles)
    map((paramfile)->rm(paramfile), paramfiles)
end

main()
