"""
Class for reading the small XML files we generate to go with each SAFE format
zipfile

"""
from __future__ import print_function, division

import os
import datetime
from xml.dom import minidom

class AusCopHubMeta(object):
    """
    Class for reading the small XML metadata files we use on the Aus 
    Copernicus Hub server, to identify and chanracterise the SAFE format
    zipfiles delivered by ESA. 
    
    Same class is used for Sentinel-1 and Sentinel-2 (and probably 3 when we get to it). 
    Not all attributes will be present, depending on the satellite. 
    
    Attributes:
        satellite:               String, e.g. S1A, S2A, etc. 
        ctrLong:                 Float, longitude of centroid of imagery
        ctrLat:                  Float, latitude of centroid of imagery
        cloudCoverPcnt:          Int, percentage cloud cover
        startTime:               datetime object, for acquisition start time (in UTC)
        stopTime:                datetime object, for acquisition stop time (in UTC)
        footprintWkt:            WKT string of rough footprint, as supplied by ESA
        esaSoftwareVersion:      String, ESA's processing software version number
        esaProcessingTimeStr:    String, time at which ESA processed (in UTC)
        polarisationValuesList:  List of strings, radar polarisation values (e.g. HV, VV, ...)
        swathValuesList:         List of strings, radar swath-type values (e.g. IW1, IW2,..)
    
    """
    def __init__(self, xmlStr=None, filename=None):
        """
        Create from either an XML string, or a filename containing the XML
        """
        if xmlStr is None and filename is not None:
            if not os.access(filename, os.R_OK):
                raise AusCopHubMetaError("XML file '{}' not found".format(filename))

            xmlStr = open(filename).read()
        if xmlStr is None:
            raise AusCopHubMetaError("Must give either xmlStr or filename argument")
        
        # Save the XML string on the object, so we can see it later on, if required. 
        self.xmlStr = xmlStr
        
        doc = minidom.parseString(xmlStr)
        
        safeDescrNodeList = doc.getElementsByTagName('AUSCOPHUB_SAFE_FILEDESCRIPTION')
        if len(safeDescrNodeList) == 0:
            raise AusCopHubMetaError("XML file '{}' is not AUSCOPHUB_SAFE_FILEDESCRIPTION file".format(filename))

        safeDescrNode = safeDescrNodeList[0]
        self.satellite = (safeDescrNode.getElementsByTagName('SATELLITE')[0]).getAttribute('name')
        centroidNodeList = safeDescrNode.getElementsByTagName('CENTROID')
        if len(centroidNodeList) > 0:
            self.ctrLong = float(centroidNodeList[0].getAttribute('longitude'))
            self.ctrLat = float(centroidNodeList[0].getAttribute('latitude'))
        cloudCoverNodeList = safeDescrNode.getElementsByTagName('ESA_CLOUD_COVER')
        if len(cloudCoverNodeList) > 0:
            self.cloudCoverPcnt = int(cloudCoverNodeList[0].getAttribute('percentage'))
        
        acqTimeNodeList = safeDescrNode.getElementsByTagName('ACQUISITION_TIME')
        if len(acqTimeNodeList) > 0:
            startTimeStr = acqTimeNodeList[0].getAttribute('start_datetime_utc')
            fmtStr = '%Y-%m-%d %H:%M:%S.%f'
            if len(startTimeStr) == 0:
                # Check for old name
                startTimeStr = acqTimeNodeList[0].getAttribute('datetime_utc')
                fmtStr = '%Y-%m-%d %H:%M:%S'
            self.startTime = datetime.datetime.strptime(startTimeStr, fmtStr)

            stopTimeStr = acqTimeNodeList[0].getAttribute('stop_datetime_utc')
            if len(stopTimeStr) > 0:
                self.stopTime = datetime.datetime.strptime(stopTimeStr, '%Y-%m-%d %H:%M:%S.%f')
        
        footprintNodeList = safeDescrNode.getElementsByTagName('ESA_TILEOUTLINE_FOOTPRINT_WKT')
        if len(footprintNodeList) > 0:
            self.footprintWkt = footprintNodeList[0].firstChild.data.strip()
        
        processingNodeList = safeDescrNode.getElementsByTagName('ESA_PROCESSING')
        if len(processingNodeList) > 0:
            self.esaSoftwareVersion = processingNodeList[0].getAttribute('software_version')
            if len(self.esaSoftwareVersion) == 0:
                self.esaSoftwareVersion = None
            self.esaProcessingTimeStr = processingNodeList[0].getAttribute('processingtime_utc')
            self.baselineCollection = processingNodeList[0].getAttribute('baselinecollection')
            if len(self.baselineCollection) == 0:
                self.baselineCollection = None
        
        polarisationNodeList = safeDescrNode.getElementsByTagName('POLARISATION')
        if len(polarisationNodeList) > 0:
            self.polarisationValuesList = polarisationNodeList[0].getAttribute('values').split(',')
        
        modeNodeList = safeDescrNode.getElementsByTagName('MODE')
        if len(modeNodeList) > 0:
            self.mode = modeNodeList[0].getAttribute('value')

        orbitNodeList = safeDescrNode.getElementsByTagName('ORBIT_NUMBERS')
        if len(orbitNodeList) > 0:
            self.relativeOrbitNumber = None
            self.absoluteOrbitNumber = None
            self.frameNumber = None
            valStr = orbitNodeList[0].getAttribute('relative')
            if len(valStr) > 0:
                self.relativeOrbitNumber = int(valStr)
            valStr = orbitNodeList[0].getAttribute('absolute')
            if len(valStr) > 0:
                self.absoluteOrbitNumber = int(valStr)
            valStr = orbitNodeList[0].getAttribute('frame')
            if len(valStr) > 0:
                self.frameNumber = int(valStr)

        passNodeList = safeDescrNode.getElementsByTagName('PASS')
        if len(passNodeList) > 0:
            self.passDirection = passNodeList[0].getAttribute('direction')

        swathNodeList = safeDescrNode.getElementsByTagName('SWATH')
        if len(swathNodeList) > 0:
            self.swathValuesList = swathNodeList[0].getAttribute('values').split(',')
        
        zipfileNode = safeDescrNode.getElementsByTagName('ZIPFILE')
        if len(zipfileNode) > 0:
            self.zipfileSizeBytes = None
            self.zipfileMd5local = None
            self.zipfileMd5esa = None
            sizeBytesStr = zipfileNode[0].getAttribute('size_bytes')
            if len(sizeBytesStr) > 0:
                self.zipfileSizeBytes = int(sizeBytesStr)
            md5local = zipfileNode[0].getAttribute('md5_local')
            if len(md5local) > 0:
                self.zipfileMd5local = md5local
            md5esa = zipfileNode[0].getAttribute('md5_esa')
            if len(md5esa) > 0:
                self.zipfileMd5esa = md5esa
        
        mgrsTileNodeList = safeDescrNode.getElementsByTagName('MGRSTILES')
        if len(mgrsTileNodeList) > 0:
            ausCopHubTilesNodeList = [node for node in mgrsTileNodeList if 
                node.getAttribute('source') == "AUSCOPHUB"]
            if len(ausCopHubTilesNodeList) > 0:
                tilesStr = ausCopHubTilesNodeList[0].firstChild.data.strip()
                self.mgrsTileList = [tile.strip() for tile in tilesStr.split('\n')]


class AusCopHubMetaError(Exception): pass
