"""
Classes for handling Sentinel-5 metadata. Initially written for Sentinel-5P,
I am guessing that this will be similar to Sentinel-5 (honestly, I don't understand
the difference).

"""
from __future__ import print_function, division

import datetime

from osgeo import gdal

from auscophub import geomutils


class Sen5Meta(object):
    """
    The metadata associated with the Sentinel-5 netCDF file.  
    
    """
    def __init__(self, ncfile=None):
        """
        Use GDAL to read the metadata dictionary
        """
        ds = gdal.Open(ncfile)
        metaDict = ds.GetMetadata()
        
        self.productType = metaDict['METADATA_GRANULE_DESCRIPTION_ProductShortName']
        startTimeStr = metaDict['time_coverage_start']
        self.startTime = datetime.datetime.strptime(startTimeStr, "%Y-%m-%dT%H:%M:%SZ")
        stopTimeStr = metaDict['time_coverage_end']
        self.stopTime = datetime.datetime.strptime(stopTimeStr, "%Y-%m-%dT%H:%M:%SZ")
        # And the generic time stamp, halfway between start and stop
        duration = self.stopTime - self.startTime
        self.datetime = self.startTime + datetime.timedelta(duration.days / 2)
        
        self.instrument = metaDict['sensor']
        self.satId = metaDict['platform']
        
        creationTimeStr = metaDict['date_created']
        self.generationTime = datetime.datetime.strptime(creationTimeStr, "%Y-%m-%dT%H:%M:%SZ")
        self.processingSoftwareVersion = metaDict['processor_version']
        # Leaving this as a string, in case they assume it later. It is a string in 
        # sen2meta. 
        self.processingLevel = metaDict['METADATA_GRANULE_DESCRIPTION_ProcessLevel']
        
        self.absoluteOrbitNumber = int(metaDict['orbit'])

        # Make an attempt at the footprint outline. Stole most of this from sen3meta. 
        # Not yet sure whether most S5P products will be swathe products, or if there
        # will be some which are chopped up further. 
        posListStr = metaDict['METADATA_EOP_METADATA_om:featureOfInterest_eop:multiExtentOf_gml:surfaceMembers_gml:exterior_gml:posList']
        posListStrVals = posListStr.split()
        numVals = len(posListStrVals)
        # Note that a gml:posList has pairs in order [lat long ....], with no sensible pair delimiter
        posListPairs = ["{} {}".format(posListStrVals[i+1], posListStrVals[i]) for i in range(0, numVals, 2)]
        posListVals = [[float(x), float(y)] for (x, y) in [pair.split() for pair in posListPairs]]

        footprintGeom = geomutils.geomFromOutlineCoords(posListVals)
        prefEpsg = geomutils.findSensibleProjection(footprintGeom)
        if prefEpsg is not None:
            self.centroidXY = geomutils.findCentroid(footprintGeom, prefEpsg)
        else:
            self.centroidXY = None
        self.outlineWKT = footprintGeom.ExportToWkt()

        # Currently have no mechanism for a preview image
        self.previewImgBin = None
        
