"""
Function to generate Sentinel-3 thumbnail
"""
from __future__ import print_function, division

import os
from distutils import spawn
import subprocess
import zipfile
import shutil

def sen3thumb(zipfilename, finalOutputDir, 
              dummy, verbose, noOverwrite, mountpath,
              pconvertpath=None, outputdir=None,
              bands=None):
    """
    Making thumbnail for Sentinel-3 using SNAP pconvert
    """
    # define bands for product
    if not bands:
        if 'OL_1' in zipfilename:
            bands ='17,6,3'       #
        elif 'SL_1_RBT' in zipfilename:
            bands = '114,110,106' #S3,S2,S1_radiance_an
        else:
            if verbose: print("Can't make thumbnail for this product.")
            return

    # confirm pconvert command
    if pconvertpath:
        cmd = os.path.join(pconvertpath, 'pconvert')
    else:
        cmd = 'pconvert'
    if not spawn.find_executable(cmd):
        raise thumbError("Executable {} is not found.".format(cmd)) 

    # confirm mount command
    # if archivemount is not available, unzip the file in the mount location
    mountcmd = 'archivemount'
    mount = True
    if not spawn.find_executable(mountcmd):
        mount = False
        if verbose:
            print("Executable {} is not found, will unzip the archive in path {}".format(mountcmd, mountpath)) 

    pngFilename = os.path.basename(zipfilename).replace('.zip', '.png')
    finalPngFile = os.path.join(finalOutputDir, pngFilename)
    if dummy:
        print("Would make", finalPngFile)
    elif os.path.exists(finalPngFile) and noOverwrite:
        if verbose:
            print("Preview image already exists {}".format(finalPngFile))
    else:
        filename = os.path.join(finalOutputDir,zipfilename)
        # make sure file exists
        if not os.path.exists(filename):
            raise thumbError("File {} is not found.".format(filename))
            
        #make the thumbnail in the file location
        if not outputdir:
            outputdir = finalOutputDir
            
        # mount the file
        mountpoint = os.path.join(mountpath, os.path.basename(zipfilename).split('.')[0])
        if verbose: print("Creating mountpoint {}.".format(mountpoint))
        os.mkdir(mountpoint)
        if mount:
            mountcmd = 'archivemount {} {}'.format(zipfilename, mountpoint)
            if verbose: print("Mounting zipfile {} to {}.".format(zipfilename, mountpoint))
            returncode = subprocess.call(mountcmd, shell=True)
            if returncode != 0:
                raise thumbError("Failed to mount file {} to point {}.".format(zipfilename, mountpoint))
        else:
            if verbose: print("Extracting zipfile {} to {}.".format(zipfilename, mountpoint))
            with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
                zip_ref.extractall(mountpoint)

        # run pconvert
        mountdir = os.listdir(mountpoint)
        if len(mountdir) != 1:
            raise thumbError("{} directories found in mountpoint {}.".format(len(mountdir), mountpoint))

        mountpath = os.path.join(mountpoint, mountdir[0])
        fullcmd = '{} -f png -r 512,512 -b {} -m equalize {} -o {}'.format(cmd, bands, mountpath, outputdir)

        # run conversion
        if verbose: print("Creating", finalPngFile)
        proc = subprocess.Popen(fullcmd, shell=True, stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr= proc.communicate()
        if proc.returncode != 0:
            if mount: umount(mountpoint)
            raise thumbError("Failed to run pconvert cmd {}; {}; {}".format(cmd, stdout, stderr))
    
        # unmount
        if mount:
            umount(mountpoint)
        shutil.rmtree(mountpoint)
        if verbose: print("Directory {} is removed.".format(mountpoint))


def umount(mountpoint):
    umountcmd = 'umount {}'.format(mountpoint)
    returncode = subprocess.call(umountcmd, shell=True)
    if returncode != 0:
        raise thumbError("Failed to unmount {}.".format(mountpoint))
    

class thumbError(Exception): pass
