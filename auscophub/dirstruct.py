"""
Utility functions for handling the storage directory structure on the AusCopernicus
Hub server. 

"""
from __future__ import print_function, division

import sys
import os
import shutil
import hashlib
import subprocess
from PIL import Image
isPython3 = (sys.version_info.major == 3)
if isPython3:
    from io import BytesIO
else:
    # Note that we are using the Python-3 name for a Python-2 class. 
    from cStringIO import StringIO as BytesIO

# Size of lat/long grid cells in which we store the files (in degrees). This is 
# potentially a function of which Sentinel we are dealing with, hence the dictionary,
# which is keyed by Sentinel number, i.e. 1, 2, 3, .....
stdGridCellSize = {
    1: 5,
    2: 5, 
    3: 40
}


# Some octal constants to use as permission modes in things like chmod()
UNIXMODE_UrwxGrwxOrx = 0o775
UNIXMODE_UrwGrOr     = 0o644
UNIXMODE_UrGrOr      = 0o444

def makeRelativeOutputDir(metainfo, gridCellSize, productDirGiven=False):
    """
    Make the output directory string for the given zipfile metadata object. The
    gridCellSize parameter is in degrees. 
    
    The productDirGiven argument is provided in order to be able to reproduce
    the old behaviour, in which the upper directory levels satellite/instrument/product
    are not generated here, but were assumed to have been given in some other way. 
    If productDirGiven is True, the result will not include these levels. 
    
    """
    satDir = makeSatelliteDir(metainfo)
    instrumentDir = makeInstrumentDir(metainfo)
    productDir = makeProductDir(metainfo)
    yearMonthDir = makeYearMonthDir(metainfo)
    dateDir = makeDateDir(metainfo)
    isOCN = False
    isWV = False # WV mode is shaped like OCN products
    # use year/month/date for OCN product
    if hasattr(metainfo, 'productType'):
        if metainfo.productType == 'OCN': isOCN = True
    if hasattr(metainfo, 'mode'):
        if metainfo.mode == 'WV': isWV = True
    if metainfo.satId[1] in ("3", "5") or isOCN:
        # For all S-3 and S-5 products we do not split spatially at all. 
        outDir = os.path.join(yearMonthDir, dateDir)
    elif isWV:
        outDir = os.path.join(yearMonthDir, 'WV')
    elif metainfo.centroidXY is not None:
        gridSquareDir = makeGridSquareDir(metainfo, gridCellSize)
        outDir = os.path.join(yearMonthDir, gridSquareDir)
    else:
        # This is a catchall fallback, just in case. 
        outDir = yearMonthDir
        
    if not productDirGiven:
        fullDir = os.path.join(satDir, instrumentDir, productDir, outDir)
    else:
        fullDir = outDir
    return fullDir
    

def makeGridSquareDir(metainfo, gridCellSize):
    """
    Make the grid square directory name, from the centroid location. Divides up
    into lat/long grid cells of the given size (given in degrees). Returns a
    string of the resulting subdirectory name
    """
    (longitude, latitude) = tuple(metainfo.centroidXY)
    i = int(latitude / gridCellSize)
    j = int(longitude / gridCellSize)
    
    longitude5left = j * gridCellSize
    if longitude < 0:
        longitude5left = longitude5left - gridCellSize
    latitude5bottom = i * gridCellSize
    if latitude < 0:
        latitude5bottom = latitude5bottom - gridCellSize
    
    # Now the top and right
    longitude5right = longitude5left + gridCellSize
    latitude5top = latitude5bottom + gridCellSize
    
    # Do we need special cases near the poles? I don't think so, but if we did, this is 
    # where we would put them, to modify the top/left/bottom/right bounds
    
    # Create the final directory string. Shows the topLeft-bottomRight coords
    dirName = "{topLat:02}{topHemi}{leftLong:03}{leftHemi}-{botLat:02}{botHemi}{rightLong:03}{rightHemi}".format(
        topLat=abs(latitude5top), topHemi=latHemisphereChar(latitude5top),
        leftLong=abs(longitude5left), leftHemi=longHemisphereChar(longitude5left),
        botLat=abs(latitude5bottom), botHemi=latHemisphereChar(latitude5bottom),
        rightLong=abs(longitude5right), rightHemi=longHemisphereChar(longitude5right))
    return dirName


def longHemisphereChar(longitude):
    """
    Appropriate hemisphere character for given longitude (i.e. "E" or "W")
    """
    return ("W" if longitude < 0 else "E")


def latHemisphereChar(latitude):
    """
    Appropriate hemisphere character for given latitude (i.e. "N" or "S")
    """
    return ("S" if latitude < 0 else "N")


def makeYearMonthDir(metainfo):
    """
    Return the string for the year/month subdirectory. The date is the acquistion date 
    of the imagery. Returns a directory structure for year/year-month, as we want to divide
    the months up a bit. After we have a few years of data, it could become rather onerous
    if we do not divide them. 
    """
    year = metainfo.startTime.year
    month = metainfo.startTime.month
    
    dirName = os.path.join("{:04}".format(year), "{:04}-{:02}".format(year, month))
    return dirName


def makeDateDir(metainfo):
    """
    Return the string for the date subdirectory. The date is the acquistion date 
    of the imagery. Returns a directory name for yyyy-mm-dd. 
    """
    year = metainfo.startTime.year
    month = metainfo.startTime.month
    day = metainfo.startTime.day
    
    dirName = "{:04}-{:02}-{:02}".format(year, month, day)
    return dirName


def makeInstrumentDir(metainfo):
    """
    Return the directory we will use at the 'instrument' level, based on the 
    metainfo object. 
    
    """
    if metainfo.satId.startswith('S1'):
        instrument = "C-SAR"
    elif metainfo.satId.startswith('S2'):
        instrument = "MSI"
    elif metainfo.satId.startswith('S3'):
        instrument = metainfo.instrument
    elif metainfo.satId.startswith('S5'):
        instrument = metainfo.instrument
    else:
        instrument = None
    return instrument


def makeSatelliteDir(metainfo):
    """
    Make the directory name for the 'satellite' level. 
    """
    satDir = "Sentinel-" + metainfo.satId[1]
    return satDir


def makeProductDir(metainfo):
    """
    Return the directory we will use at the 'product' level, based on the
    metainfo object. 
    """
    if metainfo.satId.startswith('S1'):
        product = metainfo.productType
    elif metainfo.satId.startswith('S2'):
        # Let's hope this still works when Level-2A comes along
        product = "L" + metainfo.processingLevel[-2:]
    elif metainfo.satId.startswith('S3'):
        product = metainfo.productType
    elif metainfo.satId.startswith('S5'):
        product = metainfo.productType
    else:
        product = None
    return product

def processingLevel(metainfo):
    """
    Return processing Level based on satellite and productType.
    """
    if metainfo.satId.startswith('S1'):
        if metainfo.productType in ['RAW']: return 'LEVEL-0'
        elif metainfo.productType in ['OCN']: return 'LEVEL-2'
        else: return 'LEVEL-1'
    elif metainfo.satId.startswith('S2'): return "L" + metainfo.processingLevel[-2:]
    elif metainfo.satId.startswith('S3'): return 'LEVEL-{}'.format(metainfo.processingLevel)
    elif metainfo.satId.startswith('S5'): return 'LEVEL-{}'.format(metainfo.processingLevel)
    else:
        return 'LEVEL-1'
    
def checkFinalDir(finalOutputDir, dummy, verbose):
    """
    Check that the final output dir exists, and has write permission. If it does not exist,
    then create it
    """
    exists = os.path.exists(finalOutputDir)
    if not exists:
        if dummy:
            print("Would make dir", finalOutputDir)
        else:
            if verbose:
                print("Creating dir", finalOutputDir)
            try:
                os.makedirs(finalOutputDir, UNIXMODE_UrwxGrwxOrx)   # Should the permissions come from the command line?
            except OSError as e:
                # If the error was just "File exists", then just move along, as it just means that
                # the directory was created by another process after we checked. If it was anything 
                # else then re-raise the exception, so we don't mask any other problems. 
                if "File exists" not in str(e):
                    raise 

    if not dummy:
        writeable = os.access(finalOutputDir, os.W_OK)
        if not writeable:
            raise AusCopDirStructError("Output directory {} is not writeable".format(finalOutputDir))


def moveZipfile(zipfilename, finalOutputDir, dummy, verbose, makeCopy, makeSymlink, nooverwrite,
        moveandsymlink, makereadonly=False):
    """
    Move the given zipfile to the final output directory
    """
    preExisting = False
    finalFile = os.path.join(finalOutputDir, os.path.basename(zipfilename))
    if os.path.exists(finalFile):
        if os.path.abspath(zipfilename)== os.path.abspath(finalFile):
            if verbose:
                print ("Zipfile", zipfilename, "already in final location. No move is required. ")
            preExisting = True
        elif nooverwrite:
            if verbose:
                print("Zipfile", zipfilename, "already in final location. Not moved. ")
            preExisting = True
        else:
            if dummy:
                print("Would remove pre-existing", finalFile)
            else:
                if verbose:
                    print("Removing", finalFile)
                os.remove(finalFile)

    if not preExisting:
        if dummy:
            print("Would move to", finalFile)
        else:
            if makeCopy:
                if verbose:
                    print("Copy to", finalFile)
                shutil.copyfile(zipfilename, finalFile)
                shutil.copystat(zipfilename, finalFile)
                if makereadonly: os.chmod(finalFile, UNIXMODE_UrGrOr)
            elif makeSymlink:
                if verbose:
                    print("Symlink to", finalFile)
                zipfilenameFull = os.path.abspath(zipfilename)
                os.symlink(zipfilenameFull, finalFile)
            else:
                if verbose:
                    print("Move to", finalFile)
                shutil.move(zipfilename, finalFile)
                if makereadonly: os.chmod(finalFile, UNIXMODE_UrGrOr)
                if moveandsymlink:
                    os.symlink(os.path.abspath(finalFile), os.path.abspath(zipfilename))


def createSentinel1Xml(zipfilename, finalOutputDir, metainfo, dummy, verbose, noOverwrite,
        md5esa, makereadonly=False):
    """
    Create the XML file in the final output directory, for Sentinel-1 zipfiles. 
    This is a locally-designed XML file intended to include only the sort of 
    information users would need in order to select zipfiles for download. 
    
    """
    xmlFilename = os.path.basename(zipfilename).replace('.zip', '.xml')
    finalXmlFile = os.path.join(finalOutputDir, xmlFilename)
    
    if os.path.exists(finalXmlFile):
        if noOverwrite:
            if verbose or dummy:
                print("XML already exists {}".format(finalXmlFile))
            return finalXmlFile
        else:
            if dummy:
                print("Would remove existing file {}".format(finalXmlFile)) 
            else:
                if verbose:
                    print("Removing existing file {}".format(finalXmlFile))
                os.chmod(finalXmlFile, UNIXMODE_UrwGrOr)
                os.remove(finalXmlFile)

    if dummy:
        print("Would make", finalXmlFile)        
    else:
        if verbose:
            print("Creating", finalXmlFile)
        fileInfo = ZipfileSysInfo(zipfilename)

        f = open(finalXmlFile, 'w')
        f.write("<?xml version='1.0'?>\n")
        f.write("<AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.write("  <IDENTIFIER>{}</IDENTIFIER>\n".format(os.path.basename(zipfilename).split('.')[0]))
        f.write("  <PATH>{}</PATH>\n".format(finalOutputDir.split(makeSatelliteDir(metainfo))[1]))
        f.write("  <SATELLITE name='{}' />\n".format(metainfo.satId))
        f.write("  <INSTRUMENT>{}</INSTRUMENT>\n".format("C-SAR"))
        f.write("  <PRODUCT_TYPE>{}</PRODUCT_TYPE>\n".format(metainfo.productType))  
        f.write("  <PROCESSING_LEVEL>{}</PROCESSING_LEVEL>\n".format(processingLevel(metainfo)))
        if metainfo.centroidXY is not None:
            (longitude, latitude) = tuple(metainfo.centroidXY)
            f.write("  <CENTROID longitude='{}' latitude='{}' />\n".format(longitude, latitude))
        f.write("  <ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        f.write("    {}\n".format(metainfo.outlineWKT))
        f.write("  </ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        startTimestampStr = metainfo.startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        stopTimestampStr = metainfo.stopTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        f.write("  <ACQUISITION_TIME start_datetime_utc='{}' stop_datetime_utc='{}' />\n".format(
            startTimestampStr, stopTimestampStr))
        if metainfo.polarisation is not None:
            f.write("  <POLARISATION values='{}' />\n".format(','.join(metainfo.polarisation)))
        if metainfo.swath is not None:
            f.write("  <SWATH values='{}' />\n".format(','.join(metainfo.swath)))
        f.write("  <MODE value='{}' />\n".format(metainfo.mode))
        f.write("  <ORBIT_NUMBERS relative='{}' absolute='{}' />\n".format(metainfo.relativeOrbitNumber,
            metainfo.absoluteOrbitNumber))
        if metainfo.passDirection is not None:
            f.write("  <PASS direction='{}' />\n".format(metainfo.passDirection))
            
        f.write("  <ZIPFILE size_bytes='{}' md5_local='{}' ".format(fileInfo.sizeBytes, 
            fileInfo.md5))
        if md5esa is not None:
            f.write("md5_esa='{}' ".format(md5esa.upper()))
        f.write("/>\n")
        
        f.write("</AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.close()
        if makereadonly: os.chmod(finalXmlFile, UNIXMODE_UrGrOr)
    return finalXmlFile

def createSentinel2Xml(zipfilename, finalOutputDir, metainfo, dummy, verbose, noOverwrite,
        md5esa, makereadonly=False):
    """
    Create the XML file in the final output directory, for Sentinel-2 zipfiles. 
    This is a locally-designed XML file intended to include only the sort of 
    information users would need in order to select zipfiles for download. 
    
    """
    xmlFilename = os.path.basename(zipfilename).replace('.zip', '.xml')
    finalXmlFile = os.path.join(finalOutputDir, xmlFilename)
    
    if os.path.exists(finalXmlFile):
        if noOverwrite:
            if verbose or dummy:
                print("XML already exists {}".format(finalXmlFile))
            return finalXmlFile
        else:
            if dummy:
                print("Would remove existing file {}".format(finalXmlFile)) 
            else:
                if verbose:
                    print("Removing existing file {}".format(finalXmlFile))
                os.chmod(finalXmlFile, UNIXMODE_UrwGrOr)
                os.remove(finalXmlFile)

    if dummy:
        print("Would make", finalXmlFile)
    else:
        if verbose:
            print("Creating", finalXmlFile)
        fileInfo = ZipfileSysInfo(zipfilename)
        
        f = open(finalXmlFile, 'w')
        f.write("<?xml version='1.0'?>\n")
        f.write("<AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.write("  <IDENTIFIER>{}</IDENTIFIER>\n".format(os.path.basename(zipfilename).split('.')[0]))
        f.write("  <PATH>{}</PATH>\n".format(finalOutputDir.split(makeSatelliteDir(metainfo))[1]))
        f.write("  <SATELLITE name='{}' />\n".format(metainfo.satId))
        f.write("  <INSTRUMENT>{}</INSTRUMENT>\n".format("MSI"))
        f.write("  <PRODUCT_TYPE>{}</PRODUCT_TYPE>\n".format("S2MSIL" + metainfo.processingLevel[-2:]))
        f.write("  <PROCESSING_LEVEL>{}</PROCESSING_LEVEL>\n".format(processingLevel(metainfo)))
        (longitude, latitude) = tuple(metainfo.centroidXY)
        f.write("  <CENTROID longitude='{}' latitude='{}' />\n".format(longitude, latitude))
        f.write("  <ESA_CLOUD_COVER percentage='{}' />\n".format(int(round(metainfo.cloudPcnt))))
        f.write("  <ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        f.write("    {}\n".format(metainfo.extPosWKT))
        f.write("  </ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        startTimestampStr = metainfo.startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        stopTimestampStr = metainfo.stopTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        f.write("  <ACQUISITION_TIME start_datetime_utc='{}' stop_datetime_utc='{}' />\n".format(
            startTimestampStr, stopTimestampStr))
        f.write("  <ESA_PROCESSING software_version='{}' processingtime_utc='{}'/>\n".format(
            metainfo.processingSoftwareVersion, metainfo.generationTime))
        f.write("  <ORBIT_NUMBERS relative='{}' />\n".format(metainfo.relativeOrbitNumber))
        
        f.write("  <ZIPFILE size_bytes='{}' md5_local='{}' ".format(fileInfo.sizeBytes, 
            fileInfo.md5))
        if md5esa is not None:
            f.write("md5_esa='{}' ".format(md5esa.upper()))
        f.write("/>\n")
        
        if metainfo.tileNameList is not None:
            # Only write the list of tile names if it actually exists. 
            f.write("\n")
            f.write("  <!-- These MGRS tile identifiers are not those supplied by ESA's processing software, but have been \n")
            f.write("      calculated directly from tile centroids by the Australian Copernicus Hub -->\n")
            f.write("  <MGRSTILES source='AUSCOPHUB' >\n")
            for tileName in metainfo.tileNameList:
                f.write("    {}\n".format(tileName))
            f.write("  </MGRSTILES>\n")
        f.write("</AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.close()
        if makereadonly: os.chmod(finalXmlFile, UNIXMODE_UrGrOr)
    return finalXmlFile

def createSentinel3Xml(zipfilename, finalOutputDir, metainfo, dummy, verbose, noOverwrite,
        md5esa, makereadonly=False):
    """
    Create the XML file in the final output directory, for Sentinel-3 zipfiles. 
    This is a locally-designed XML file intended to include only the sort of 
    information users would need in order to select zipfiles for download. 
    
    """
    xmlFilename = os.path.basename(zipfilename).replace('.zip', '.xml')
    finalXmlFile = os.path.join(finalOutputDir, xmlFilename)
    
    if os.path.exists(finalXmlFile):
        if noOverwrite:
            if verbose or dummy:
                print("XML already exists {}".format(finalXmlFile))
            return finalXmlFile
        else:
            if dummy:
                print("Would remove existing file {}".format(finalXmlFile)) 
            else:
                if verbose:
                    print("Removing existing file {}".format(finalXmlFile))
                os.chmod(finalXmlFile, UNIXMODE_UrwGrOr)
                os.remove(finalXmlFile)

    if dummy:
        print("Would make", finalXmlFile)
    else:
        if verbose:
            print("Creating", finalXmlFile)
        fileInfo = ZipfileSysInfo(zipfilename)
        
        f = open(finalXmlFile, 'w')
        f.write("<?xml version='1.0'?>\n")
        f.write("<AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.write("  <IDENTIFIER>{}</IDENTIFIER>\n".format(os.path.basename(zipfilename).split('.')[0]))
        f.write("  <PATH>{}</PATH>\n".format(finalOutputDir.split(makeSatelliteDir(metainfo))[1]))
        f.write("  <SATELLITE name='{}' />\n".format(metainfo.satId))
        f.write("  <INSTRUMENT>{}</INSTRUMENT>\n".format(metainfo.instrument))
        f.write("  <PRODUCT_TYPE>{}</PRODUCT_TYPE>\n".format(metainfo.productType))  
        f.write("  <PROCESSING_LEVEL>{}</PROCESSING_LEVEL>\n".format(processingLevel(metainfo)))
        if metainfo.centroidXY is not None:
            (longitude, latitude) = tuple(metainfo.centroidXY)
            f.write("  <CENTROID longitude='{}' latitude='{}' />\n".format(longitude, latitude))
        f.write("  <ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        f.write("    {}\n".format(metainfo.outlineWKT))
        f.write("  </ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        startTimestampStr = metainfo.startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        stopTimestampStr = metainfo.stopTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        f.write("  <ACQUISITION_TIME start_datetime_utc='{}' stop_datetime_utc='{}' />\n".format(
            startTimestampStr, stopTimestampStr))
        f.write("  <ESA_PROCESSING processingtime_utc='{}' baselinecollection='{}'/>\n".format(
            metainfo.generationTime, metainfo.baselineCollection))
        f.write("  <ORBIT_NUMBERS relative='{}' ".format(metainfo.relativeOrbitNumber))
        if metainfo.frameNumber is not None:
            f.write("frame='{}' ".format(metainfo.frameNumber))
        if metainfo.absoluteOrbitNumber is not None:
            f.write("absolute='{}' ".format(metainfo.absoluteOrbitNumber))
        if metainfo.cycleNumber is not None:
            f.write("cycle='{}' ".format(metainfo.cycleNumber))
        f.write("/>\n")
        
        f.write("  <ZIPFILE size_bytes='{}' md5_local='{}' ".format(fileInfo.sizeBytes, 
            fileInfo.md5))
        if md5esa is not None:
            f.write("md5_esa='{}' ".format(md5esa.upper()))
        f.write("/>\n")
        
        f.write("</AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.close()
        if makereadonly: os.chmod(finalXmlFile, UNIXMODE_UrGrOr)
    return finalXmlFile


def createSentinel5Xml(ncfilename, finalOutputDir, metainfo, dummy, verbose, noOverwrite,
        md5esa, makereadonly=False):
    """
    Create the XML file in the final output directory, for Sentinel-5 netCDF files. 
    This is a locally-designed XML file intended to include only the sort of 
    information users would need in order to select ncfiles for download. 
    
    Note that I have left the top-level XML tag as <AUSCOPHUB_SAFE_FILEDESCRIPTION>,
    even though technically this is not a SAFE format file, because there is a lot of 
    other stuff which assumes this down the track, and it was not worth changing it. I 
    should have used a more generic tag name in the first place (with hindsight). 
    
    """
    xmlFilename = os.path.basename(ncfilename).replace('.nc', '.xml')
    finalXmlFile = os.path.join(finalOutputDir, xmlFilename)
    
    if os.path.exists(finalXmlFile):
        if noOverwrite:
            if verbose or dummy:
                print("XML already exists {}".format(finalXmlFile))
            return finalXmlFile
        else:
            if dummy:
                print("Would remove existing file {}".format(finalXmlFile)) 
            else:
                if verbose:
                    print("Removing existing file {}".format(finalXmlFile))
                os.chmod(finalXmlFile, UNIXMODE_UrwGrOr)
                os.remove(finalXmlFile)

    if dummy:
        print("Would make", finalXmlFile)
    else:
        if verbose:
            print("Creating", finalXmlFile)
        fileInfo = ZipfileSysInfo(ncfilename)
        
        f = open(finalXmlFile, 'w')
        f.write("<?xml version='1.0'?>\n")
        f.write("<AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.write("  <IDENTIFIER>{}</IDENTIFIER>\n".format(os.path.basename(ncfilename).split('.')[0]))
        f.write("  <PATH>{}</PATH>\n".format(finalOutputDir.split(makeSatelliteDir(metainfo))[1]))
        f.write("  <SATELLITE name='{}' />\n".format(metainfo.satId))
        f.write("  <INSTRUMENT>{}</INSTRUMENT>\n".format(metainfo.instrument))
        f.write("  <PRODUCT_TYPE>{}</PRODUCT_TYPE>\n".format(metainfo.productType))  
        f.write("  <PROCESSING_LEVEL>{}</PROCESSING_LEVEL>\n".format(processingLevel(metainfo)))
        if metainfo.centroidXY is not None:
            (longitude, latitude) = tuple(metainfo.centroidXY)
            f.write("  <CENTROID longitude='{}' latitude='{}' />\n".format(longitude, latitude))
        f.write("  <ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        f.write("    {}\n".format(metainfo.outlineWKT))
        f.write("  </ESA_TILEOUTLINE_FOOTPRINT_WKT>\n")
        startTimestampStr = metainfo.startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        stopTimestampStr = metainfo.stopTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        f.write("  <ACQUISITION_TIME start_datetime_utc='{}' stop_datetime_utc='{}' />\n".format(
            startTimestampStr, stopTimestampStr))
        f.write("  <ESA_PROCESSING processingtime_utc='{}' software_version='{}'/>\n".format(
            metainfo.generationTime, metainfo.processingSoftwareVersion))
        f.write("  <ORBIT_NUMBERS absolute='{}' ".format(metainfo.absoluteOrbitNumber))
        f.write("/>\n")
        
        f.write("  <ZIPFILE size_bytes='{}' md5_local='{}' ".format(fileInfo.sizeBytes, 
            fileInfo.md5))
        if md5esa is not None:
            f.write("md5_esa='{}' ".format(md5esa.upper()))
        f.write("/>\n")
        
        f.write("</AUSCOPHUB_SAFE_FILEDESCRIPTION>\n")
        f.close()
        if makereadonly: os.chmod(finalXmlFile, UNIXMODE_UrGrOr)
    return finalXmlFile


def createPreviewImg(zipfilename, finalOutputDir, metainfo, dummy, verbose, noOverwrite, makereadonly=False):
    """
    Create the preview image, in the final output directory
    """
    pngFilename = os.path.basename(zipfilename).replace('.zip', '.png')
    finalPngFile = os.path.join(finalOutputDir, pngFilename)
    
    if metainfo.previewImgBin is None:
        if verbose or dummy:
            print("No preview image provided in", zipfilename)
        return
    elif os.path.exists(finalPngFile):
        if noOverwrite:
            if verbose or dummy:
                print("Preview image already exists {}".format(finalPngFile))
            return
        else:
            if dummy:
                print("Would remove existing file {}".format(finalPngFile)) 
            else:
                if verbose:
                    print("Removing existing file {}".format(finalPngFile))
                os.chmod(finalPngFile, UNIXMODE_UrwGrOr)
                os.remove(finalPngFile) 

    if dummy:
        print("Would make", finalPngFile)
    else:
        if verbose:
            print("Creating", finalPngFile)

        #resize the image
        qldata = BytesIO(metainfo.previewImgBin)
        im = Image.open(qldata)
        im.thumbnail((512,512), Image.ANTIALIAS)
        if os.path.basename(zipfilename).startswith('S1'):
            # preview always has top-left as first sensing pixel. 
            # flip according to orbit direction.
            if metainfo.passDirection.lower().startswith('asc'):
                if verbose: print("Flipping preview top-bottom")
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                if verbose: print("Flipping preview left-right")
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.save(finalPngFile, "PNG")
        if makereadonly: os.chmod(finalPngFile, UNIXMODE_UrGrOr)


class ZipfileSysInfo(object):
    """
    Information about the zipfile which can be obtained at operating system level,
    without understanding the internal structure of the zipfile (i.e. it is just
    a file). 
    
    """
    def __init__(self, zipfilename):
        statInfo = os.stat(zipfilename)
        self.sizeBytes = statInfo.st_size
        self.md5 = self.md5hash(zipfilename).upper()
    
    @staticmethod
    def md5hash(zipfilename):
        """
        Calculate the md5 hash of the given zipfile
        """
        hashObj = hashlib.md5()
        blocksize = 65536
        f = open(zipfilename, 'rb')
        buf = f.read(blocksize)
        while len(buf) > 0:
            hashObj.update(buf)
            buf = f.read(blocksize)
        return hashObj.hexdigest()


class AusCopDirStructError(Exception): pass

