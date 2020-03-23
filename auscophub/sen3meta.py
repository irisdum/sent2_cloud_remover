"""
Classes for handling Sentinel-3 metadata
"""
from __future__ import print_function, division

import zipfile
import datetime
from xml.dom import minidom

from osgeo import ogr

from auscophub import geomutils


class Sen3ZipfileMeta(object):
    """
    The metadata associated with the SAFE format file. The metadata is contained
    within a single XML file, inside the SAFE directory. The
    constructor for this class takes an XML string which has been read from 
    that file, or the name of the XML file, or the name of the zipped
    SAFE file. In the latter case, the XML file will be read directly
    from the zipfile. 
    
    """
    def __init__(self, xmlStr=None, xmlfilename=None, zipfilename=None):
        if xmlStr is None:
            if xmlfilename is not None:
                xmlStr = open(xmlfilename).read()
            elif zipfilename is not None:
                zf = zipfile.ZipFile(zipfilename, 'r')
                filenames = [zi.filename for zi in zf.infolist()]
                metadataXmlfile = [fn for fn in filenames if fn.endswith('xfdumanifest.xml')][0]
                mf = zf.open(metadataXmlfile)
                xmlStr = mf.read()
                del mf
        
        doc = minidom.parseString(xmlStr)

        xfduNode = doc.getElementsByTagName('xfdu:XFDU')[0]
        metadataSectionNode = xfduNode.getElementsByTagName('metadataSection')[0]
        metadataNodeList = metadataSectionNode.getElementsByTagName('metadataObject')
        
        # Acquisition times
        acquisitionPeriodNode = self.findMetadataNodeByIdName(metadataNodeList, 'acquisitionPeriod')
        startTimeNode = acquisitionPeriodNode.getElementsByTagName('sentinel-safe:startTime')[0]
        startTimeStr = startTimeNode.firstChild.data.strip()
        self.startTime = datetime.datetime.strptime(startTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        stopTimeNode = acquisitionPeriodNode.getElementsByTagName('sentinel-safe:stopTime')[0]
        stopTimeStr = stopTimeNode.firstChild.data.strip()
        self.stopTime = datetime.datetime.strptime(stopTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Platform details
        platformNode = self.findMetadataNodeByIdName(metadataNodeList, 'platform')
        familyNameNodeList = platformNode.getElementsByTagName('sentinel-safe:familyName') 
        satFamilyNameNode = [node for node in familyNameNodeList
            if node.getAttribute('abbreviation') == ''][0]
        satFamilyNameStr = satFamilyNameNode.firstChild.data.strip()
        satNumberNode = platformNode.getElementsByTagName('sentinel-safe:number')[0]
        satNumberStr = satNumberNode.firstChild.data.strip()
        if satFamilyNameStr == "Sentinel-3":
            self.satId = "S3" + satNumberStr
        else:
            raise Sen3MetaError("Satellite family = '{}', does not appear to be Sentinel-3".format(
                satFamilyNameStr))
        instrumentNode = platformNode.getElementsByTagName('sentinel-safe:instrument')[0]
        instrFamilyNameNode = instrumentNode.getElementsByTagName('sentinel-safe:familyName')[0]
        self.instrument = instrFamilyNameNode.getAttribute('abbreviation')

        # Footprint. Confusingly, this is stored under the measurementFrameSet metadata node. 
        frameSetNode = self.findMetadataNodeByIdName(metadataNodeList, 'measurementFrameSet')
        posListNode = frameSetNode.getElementsByTagName('gml:posList')[0]
        posListStr = posListNode.firstChild.data.strip()
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

        # Frame, which is not stored in the measurementFrameSet node, but in 
        # the generalProductInfo node. 
        prodInfoNode = self.findMetadataNodeByIdName(metadataNodeList, 'generalProductInformation')
        frameNodeList = prodInfoNode.getElementsByTagName('sentinel3:alongtrackCoordinate')
        self.frameNumber = None
        if len(frameNodeList) > 0:
            frameNode = frameNodeList[0]
            self.frameNumber = int(frameNode.firstChild.data.strip())
        
        # Processing level
        productTypeNode = prodInfoNode.getElementsByTagName('sentinel3:productType')[0]
        self.productType = productTypeNode.firstChild.data.strip()
        self.processingLevel = self.productType[3]
        self.productName = self.productType[5:]
        
        # Product creation/processing time. Note that they use a different time format (sigh.....)
        creationTimeNode = prodInfoNode.getElementsByTagName('sentinel3:creationTime')[0]
        generationTimeStr = creationTimeNode.firstChild.data.strip()
        self.generationTime = datetime.datetime.strptime(generationTimeStr, "%Y%m%dT%H%M%S")
        # I think this is as close as we get to a software version number. 
        baselineNode = prodInfoNode.getElementsByTagName('sentinel3:baselineCollection')[0]
        self.baselineCollection = baselineNode.firstChild.data.strip()

        # Orbit number
        orbitRefNode = self.findMetadataNodeByIdName(metadataNodeList, 'measurementOrbitReference')
        relativeOrbitNode = orbitRefNode.getElementsByTagName('sentinel-safe:relativeOrbitNumber')[0]
        self.relativeOrbitNumber = int(relativeOrbitNode.firstChild.data.strip())
        absoluteOrbitNode = orbitRefNode.getElementsByTagName('sentinel-safe:orbitNumber')[0]
        self.absoluteOrbitNumber = int(absoluteOrbitNode.firstChild.data.strip())
        cycleNode = orbitRefNode.getElementsByTagName('sentinel-safe:cycleNumber')[0]
        self.cycleNumber = int(cycleNode.firstChild.data.strip())
        
        # MD5 checksum for .nc files
        dataSectionNode = xfduNode.getElementsByTagName('dataObjectSection')[0]
        dataList = dataSectionNode.getElementsByTagName('dataObject')
        md5={}
        for dataObject in dataList:
            key=dataObject.getElementsByTagName('fileLocation')[0].getAttribute('href')
            value=dataObject.getElementsByTagName('checksum')[0].firstChild.data.strip()
            md5[key]=value
        self.md5=md5

        # Currently have no mechanism for a preview image
        self.previewImgBin = None
        
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


class Sen3MetaError(Exception): pass
