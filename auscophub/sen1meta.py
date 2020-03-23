"""
Classes for handling Sentinel-1 metadata
"""
from __future__ import print_function, division

import os
import zipfile
import datetime
from xml.dom import minidom

from auscophub import geomutils


class Sen1ZipfileMeta(object):
    """
    This class is designed to operate on the whole zipfile of a Sentinel-1 SAFE dataset. 
    
    This version uses the top level manifest.safe file.

    Additional metadata can be found within the SAFE archive.    
    Someone who knows more about radar than me should work on classes to handle the individual
    files, when more detail is required. 
    
    """
    def __init__(self, zipfilename=None):
        """
        Currently only operates on the zipfile itself. 
        """
        if zipfilename is None:
            raise Sen1MetaError("Must give zipfilename")
        
        zf = zipfile.ZipFile(zipfilename, 'r')
        filenames = [zi.filename for zi in zf.infolist()]
        safeDirName = [fn for fn in filenames if fn.endswith('.SAFE/')][0]
        bn = safeDirName.replace('.SAFE/', '')
        
        #use manifest.safe
        metafilename = 'manifest.safe'
        fullmetafilename = safeDirName + metafilename
        mf = zf.open(fullmetafilename)
        xmlStr = mf.read()
        del mf

        doc = minidom.parseString(xmlStr)
        xfduNode = doc.getElementsByTagName('xfdu:XFDU')[0]
        metadataSectionNode = xfduNode.getElementsByTagName('metadataSection')[0]
        metadataNodeList = metadataSectionNode.getElementsByTagName('metadataObject')

        # Product information
        generalProductInformation = self.findMetadataNodeByIdName(metadataNodeList, 'generalProductInformation')
        productInformation = self.getElementsContainTagName(generalProductInformation,'standAloneProductInformation')[0]
        
        productTypeNodes=self.getElementsContainTagName(productInformation,'productType')
        if len(productTypeNodes)>0:
            self.productType= productTypeNodes[0].firstChild.data.strip()
        else:
            #this may happen if product type is RAW
            self.productType= os.path.basename(zipfilename).split("_")[2]
            
        self.polarisation = sorted([node.firstChild.data.strip() for node in self.getElementsContainTagName(productInformation,'transmitterReceiverPolarisation')])

        #productInformation = generalProductInformation.getElementsByTagName('s1sarl1:standAloneProductInformation')[0]
        #self.productType = productInformation.getElementsByTagName('s1sarl1:productType')[0].firstChild.data.strip()
        #self.polarisation = sorted([node.firstChild.data.strip() for node in productInformation.getElementsByTagName('s1sarl1:transmitterReceiverPolarisation')])
        
        # Acquisition times
        acquisitionPeriodNode = self.findMetadataNodeByIdName(metadataNodeList, 'acquisitionPeriod')
        startTimeNode = self.getElementsContainTagName(acquisitionPeriodNode,'startTime')[0]
        startTimeStr = startTimeNode.firstChild.data.strip()
        if 'Z' in startTimeStr[-1]:
            self.startTime = datetime.datetime.strptime(startTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            self.startTime = datetime.datetime.strptime(startTimeStr, "%Y-%m-%dT%H:%M:%S.%f")
        stopTimeNode = self.getElementsContainTagName(acquisitionPeriodNode,'stopTime')[0]
        stopTimeStr = stopTimeNode.firstChild.data.strip()
        if 'Z' in startTimeStr[-1]:
            self.stopTime = datetime.datetime.strptime(stopTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        else: 
            self.stopTime = datetime.datetime.strptime(stopTimeStr, "%Y-%m-%dT%H:%M:%S.%f")

        # platform
        platform = self.findMetadataNodeByIdName(metadataNodeList, 'platform')
        platformNode = self.getElementsContainTagName(platform,'platform')[0]
        familyName = self.getElementsContainTagName(platformNode,'familyName')[0].firstChild.data.strip()
        # to be consistent with other metadata, this has to be "S1" not "Sentinel-1"
        self.satId = familyName[0]+ familyName[-1]+self.getElementsContainTagName(platformNode,'number')[0].firstChild.data.strip()
        
        instrumentMode = self.getElementsContainTagName(platform,'instrumentMode')[0]
        self.mode = self.getElementsContainTagName(instrumentMode,'mode')[0].firstChild.data.strip()
        self.swath = sorted([node.firstChild.data.strip() for node in self.getElementsContainTagName(instrumentMode,'swath')])
        
        # orbit
        measurementOrbitReference = self.findMetadataNodeByIdName(metadataNodeList, 'measurementOrbitReference')
        orbitReference = self.getElementsContainTagName(measurementOrbitReference,'orbitReference')[0]
        self.absoluteOrbitNumber = self.getElementsContainTagName(orbitReference,'orbitNumber')[0].firstChild.data.strip()
        self.relativeOrbitNumber = self.getElementsContainTagName(orbitReference,'relativeOrbitNumber')[0].firstChild.data.strip()
        self.passDirection = measurementOrbitReference.getElementsByTagName('s1:orbitProperties')[0].getElementsByTagName('s1:pass')[0].firstChild.data.strip().title()


        # footprint
        measurementFrameSet = self.findMetadataNodeByIdName(metadataNodeList, 'measurementFrameSet')      
        posSet = self.getElementsContainTagName(measurementFrameSet,'coordinates')
        # first footprint
        posListStr =  posSet[0].firstChild.data.strip()
        # This list has pairs in order [lat,long lat,long....], different from S1 and S3
        posListPairs= posListStr.split()
        posListVals = [[float(y), float(x)] for (x, y) in [pair.split(',') for pair in posListPairs]]
        footprintGeom = geomutils.geomFromOutlineCoords(posListVals)
        footprintGeom.CloseRings()
        
        # there are more than one polygons for WV products
        if len(posSet)>1:
            footprintGeom = geomutils.ogr.ForceToMultiPolygon(footprintGeom)
            for pos in posSet[1:]:
                posListPairs= pos.firstChild.data.strip().split()
                posListVals = [[float(y), float(x)] for (x, y) in [pair.split(',') for pair in posListPairs]]
                footprint = geomutils.geomFromOutlineCoords(posListVals)
                footprint.CloseRings()
                footprintGeom.AddGeometry(footprint)
            #footprintGeom=footprintGeom.ConvexHull()
        
        self.centroidXY = None
        if footprintGeom.GetGeometryName().upper() == 'POLYGON':
            prefEpsg = geomutils.findSensibleProjection(footprintGeom)
            if prefEpsg is not None:
                self.centroidXY = geomutils.findCentroid(footprintGeom, prefEpsg)
        self.outlineWKT = footprintGeom.ExportToWkt()
                
        # Grab preview data if available, for making a quick-look
        previewDir = os.path.join(safeDirName, "preview")
        previewImgFiles = [fn for fn in filenames if os.path.dirname(fn) == previewDir and
            fn.endswith('.png')]
        self.previewImgBin = None
        if len(previewImgFiles) > 0:
            # If we found some preview images, use the first one. In fact there is probably 
            # only one
            try:
                pf = zf.open(previewImgFiles[0])
                self.previewImgBin = pf.read()
                del pf
            except zipfile.BadZipfile:
                pass
        
        if not hasattr(self, 'startTime'):
            # Assume we could not find anything from inside the zipfile, so
            # fall back to things we can deduce from the filename
            self.fallbackMetadataFromFilename(zipfilename)
    
    
    @staticmethod
    def findMetadataNodeByIdName(metadataNodeList, idName):
        """
        Search the given list of metadataNode objects, and find the first one with the
        given ID=idName. If no such object found, return None.

        """
        metadataObj = None
        matchingNodes = [node for node in metadataNodeList if node.getAttribute('ID') == idName]
        if len(matchingNodes) > 0:
            metadataObj = matchingNodes[0]

        return metadataObj
    
            
    @staticmethod
    def getElementsContainTagName(node, tagname):
        """
        Search in a node based on partial tagname. This has to do with the inconsistent naming for different product levels.
        """
        elementList=[]
        for child in node.childNodes:
            if hasattr(child,'tagName'):
                if tagname in child.tagName:
                    elementList.append(child)
                else:
                    elementList.extend(Sen1ZipfileMeta.getElementsContainTagName(child,tagname))
        return elementList
            

    def fallbackMetadataFromFilename(self, zipfilename):
        """
        This is called for the product types which we do not really know how to handle, e.g.
        RAW and OCN. It fills in a rudimentary amount of metadata which can be gleaned
        directly from the zipfile name itself. 
        
        We really should do more to understand the internals of these other products, 
        particularly the OCN, which is probably manageable, and useful. The RAW perhaps not 
        so much. 
        
        """
        filenamePieces = os.path.basename(zipfilename).replace(".zip", "").split("_")
        
        self.centroidXY = None
        self.polarisation = None
        self.swath = None
        self.passDirection = None
        self.satellite = filenamePieces[0]
        self.satId = self.satellite
        self.mode = filenamePieces[1]
        self.productType = filenamePieces[2]
        if self.productType in ('RAW', 'OCN'):
            startTimeStr = filenamePieces[5]
            self.startTime = datetime.datetime.strptime(startTimeStr, "%Y%m%dT%H%M%S")
            stopTimeStr = filenamePieces[6]
            self.stopTime = datetime.datetime.strptime(stopTimeStr, "%Y%m%dT%H%M%S")
            
            self.absoluteOrbitNumber = int(filenamePieces[7])
            self.relativeOrbitNumber()
        

class Sen1MetaError(Exception): pass
