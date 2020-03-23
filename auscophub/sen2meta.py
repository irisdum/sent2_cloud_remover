"""
Classes for handling the various metadata files which come with Sentinel-2

"""
from __future__ import print_function, division

import datetime
from xml.dom import minidom
import zipfile
import fnmatch

import numpy
from osgeo import osr
from osgeo import ogr

from auscophub import geomutils

try:
    # If we have the QVF module available, we will be able to make QVF filenames, but not otherwise. 
    import qvf
except ImportError:
    qvf = None

class Sen2TileMeta(object):
    """
    Metadata for a single 100km tile
    """
    def __init__(self, filename=None, fileobj=None):
        """
        Constructor takes either a filename or a fileobj which points
        to an already opened file. The latter allows us to read from 
        a zip archive of multiple files. 
        
        """
        if filename is not None:
            f = open(filename)
        elif fileobj is not None:
            f = fileobj
        else:
            raise Sen2MetaError("Must give either filename or fileobj")
        
        xmlStr = f.read()
        doc = minidom.parseString(xmlStr)
        
        generalInfoNode = doc.getElementsByTagName('n1:General_Info')[0]
        # N.B. I am still not entirely convinced that this SENSING_TIME is really 
        # the acquisition time, but the documentation is rubbish. 
        sensingTimeStr = findElementByXPath(generalInfoNode, 'SENSING_TIME')[0].firstChild.data.strip()
        self.datetime = datetime.datetime.strptime(sensingTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        tileIdNode = findElementByXPath(generalInfoNode, 'TILE_ID')[0].firstChild
        tileIdFullStr = tileIdNode.data.strip()
        # This is the tile id as given by ESA. However, it turns out that they have 
        # got these wrong in earlier versions, so we may fix this later, after we have all the 
        # location information loaded. See self.tileId further down. 
        self.tileId_esa = tileIdFullStr.split('_')[-2]
        self.satId = tileIdFullStr[:3]
        self.procLevel = tileIdFullStr[13:16]    # Not sure whether to use absolute pos or split by '_'....
        self.processingSoftwareVersion = tileIdFullStr[-5:]
        
        geomInfoNode = doc.getElementsByTagName('n1:Geometric_Info')[0]
        geocodingNode = findElementByXPath(geomInfoNode, 'Tile_Geocoding')[0]
        epsgNode = findElementByXPath(geocodingNode, 'HORIZONTAL_CS_CODE')[0]
        self.epsg = epsgNode.firstChild.data.strip().split(':')[1]
        
        # Dimensions of images at different resolutions. 
        self.dimsByRes = {}
        sizeNodeList = findElementByXPath(geocodingNode, 'Size')
        for sizeNode in sizeNodeList:
            res = sizeNode.getAttribute('resolution')
            nrows = int(findElementByXPath(sizeNode, 'NROWS')[0].firstChild.data.strip())
            ncols = int(findElementByXPath(sizeNode, 'NCOLS')[0].firstChild.data.strip())
            self.dimsByRes[res] = (nrows, ncols)

        # Upper-left corners of images at different resolutions. As far as I can
        # work out, these coords appear to be the upper left corner of the upper left
        # pixel, i.e. equivalent to GDAL's convention. This also means that they
        # are the same for the different resolutions, which is nice. 
        self.ulxyByRes = {}
        posNodeList = findElementByXPath(geocodingNode, 'Geoposition')
        for posNode in posNodeList:
            res = posNode.getAttribute('resolution')
            ulx = float(findElementByXPath(posNode, 'ULX')[0].firstChild.data.strip())
            uly = float(findElementByXPath(posNode, 'ULY')[0].firstChild.data.strip())
            self.ulxyByRes[res] = (ulx, uly)
        
        # Our own version of the tile id, which we will use as the scene name
        self.tileId = self.tileId_esa.lower()
        if self.processingSoftwareVersion < "02.01":
            # ESA had a bug in their MGRS algorithm, fixed in version 02.01. If the version
            # is earlier, check against our own calculation, from the tile centroid. 
            mgrsStr = 't' + self.calcMGRSname().lower()
            # The bug apparently only affects the final character, so only change that. 
            if mgrsStr[-1] != self.tileId[-1]:
                if mgrsStr[:-1] == self.tileId[:-1]:
                    self.tileId = mgrsStr
        
        # Sun and satellite angles. 
        tileAnglesNode = findElementByXPath(geomInfoNode, 'Tile_Angles')[0]
        self.angleGridXres = float(findElementByXPath(tileAnglesNode, 'Sun_Angles_Grid/Zenith/COL_STEP')[0].firstChild.data)
        self.angleGridYres = float(findElementByXPath(tileAnglesNode, 'Sun_Angles_Grid/Zenith/ROW_STEP')[0].firstChild.data)
        self.sunZenithGrid = self.makeValueArray(findElementByXPath(tileAnglesNode, 'Sun_Angles_Grid/Zenith/Values_List')[0])
        self.sunAzimuthGrid = self.makeValueArray(findElementByXPath(tileAnglesNode, 'Sun_Angles_Grid/Azimuth/Values_List')[0])
        self.anglesGridShape = self.sunAzimuthGrid.shape
        
        # Now build up the viewing angle per grid cell, from the separate layers
        # given for each detector for each band. Initially I am going to keep
        # the bands separate, just to see how that looks. 
        # The names of things in the XML suggest that these are view angles,
        # but the numbers suggest that they are angles as seen from the pixel's 
        # frame of reference on the ground, i.e. they are in fact what we ultimately want. 
        viewingAngleNodeList = findElementByXPath(tileAnglesNode, 'Viewing_Incidence_Angles_Grids')
        self.viewZenithDict = self.buildViewAngleArr(viewingAngleNodeList, 'Zenith')
        self.viewAzimuthDict = self.buildViewAngleArr(viewingAngleNodeList, 'Azimuth')
        
        # Make a guess at the coordinates of the angle grids. These are not given 
        # explicitly in the XML, and don't line up exactly with the other grids, so I am 
        # making a rough estimate. Because the angles don't change rapidly across these 
        # distances, it is not important if I am a bit wrong (although it would be nice
        # to be exactly correct!). 
        (ulx, uly) = self.ulxyByRes["10"]
        self.anglesULXY = (ulx - self.angleGridXres / 2.0, uly + self.angleGridYres / 2.0)
    
    @staticmethod
    def makeValueArray(valuesListNode):
        """
        Take a <Values_List> node from the XML, and return an array of the values contained
        within it. This will be a 2-d numpy array of float32 values (should I pass the dtype in??)
        
        """
        valuesList = findElementByXPath(valuesListNode, 'VALUES')
        vals = []
        for valNode in valuesList:
            text = valNode.firstChild.data.strip()
            vals.append([numpy.float32(x) for x in text.split()])
        return numpy.array(vals)
    
    def buildViewAngleArr(self, viewingAngleNodeList, angleName):
        """
        Build up the named viewing angle array from the various detector strips given as
        separate arrays. I don't really understand this, and may need to re-write it once
        I have worked it out......
        
        The angleName is one of 'Zenith' or 'Azimuth'.
        Returns a dictionary of 2-d arrays, keyed by the bandId string. 
        """
        angleArrDict = {}
        for viewingAngleNode in viewingAngleNodeList:
            bandId = viewingAngleNode.getAttribute('bandId')
            angleNode = findElementByXPath(viewingAngleNode, angleName)[0]
            angleArr = self.makeValueArray(findElementByXPath(angleNode, 'Values_List')[0])
            if bandId not in angleArrDict:
                angleArrDict[bandId] = angleArr
            else:
                mask = (~numpy.isnan(angleArr))
                angleArrDict[bandId][mask] = angleArr[mask]
        return angleArrDict

    def getUTMzone(self):
        """
        Return the UTM zone of the tile, as an integer
        """
        if not (self.epsg.startswith("327") or self.epsg.startswith("326")):
            raise Sen2MetaError("Cannot determine UTM zone from EPSG:{}".format(self.epsg))
        return int(self.epsg[3:])
    
    def getCtrXY(self):
        """
        Return the (X, Y) coordinates of the scene centre (in image projection, generally UTM)
        """
        (nrows, ncols) = self.dimsByRes['10']
        (ctrRow, ctrCol) = (nrows // 2, ncols // 2)
        (ulx, uly) = self.ulxyByRes['10']
        (ctrX, ctrY) = (ulx + ctrCol * 10, uly - ctrRow * 10)
        return (ctrX, ctrY)
    
    def getCtrLongLat(self):
        """
        Return the (longitude, latitude) of the scene centre
        """
        (ctrX, ctrY) = self.getCtrXY()
        srUTM = osr.SpatialReference()
        srUTM.ImportFromEPSG(int(self.epsg))
        srLL = osr.SpatialReference()
        srLL.ImportFromEPSG(4326)
        tr = osr.CoordinateTransformation(srUTM, srLL)
        (longitude, latitude, z) = tr.TransformPoint(ctrX, ctrY)
        return (longitude, latitude)

    def makeQVFname(self):
        """
        Make a QVF name for the current tile. Stage is aa0 and suffix is .meta,
        but these are just notional placeholders. 
        
        """
        if qvf is None:
            raise Sen2MetaError("Cannot make QVF names, as qvf module is unavailable")

        satIdDict = {'S2A':'ce', 'S2B':'cf', 'S2C':'cg', 'S2D':'ch'}
        
        sat = satIdDict[self.satId]
        instr = "ms"
        prod = "re"
        stage = "aa0"
        utmZone = self.getUTMzone()
        if utmZone >= 50 and utmZone < 60:
            zoneCode = "m{}".format(utmZone-50)
        else:
            zoneCode = "m{:02}".format(zoneCode)

        what = sat+instr+prod
        where = self.tileId.lower()
        date = self.datetime.strftime("%Y%m%d")
        baseQVFname = qvf.assemble([what, where, date, stage+zoneCode])
        baseQVFname = qvf.changesuffix(baseQVFname, 'meta')
        return baseQVFname
    
    def calcMGRSname(self):
        """
        Calculate the US-MGRS 100km grid square identifier. This should not be necessary,
        as ESA have supplied the tile name in the granule names, but they appear to have 
        got it wrong sometimes. So, in order to make sure we have it right, I calculate it 
        myself. How very tiresome. 

        """
        epsg = self.epsg
        (easting, northing) = self.getCtrXY()
        
        mgrsString = calcMGRSnameFromCoords(epsg, easting, northing)
        return mgrsString


class Sen2ZipfileMeta(object):
    """
    The metadata associated with the SAFE format file, which is a collection of 
    multiple tiles. At the top of the SAFE hierarchy is a single XML file with metadata
    which applies to the whole collection, and this class carries information from 
    that XML file. The constructor can take either the XML as a string or the name of
    an XML file, or the name of a zipped SAFE file. 
    
    """
    def __init__(self, xmlStr=None, xmlfilename=None, zipfilename=None):
        """
        Take either the name of a zipfile, an XML file, or an XML string, and construct
        the object from the metadata
        """
        self.previewImgBin = None
        if xmlStr is None:
            if xmlfilename is not None:
                xmlStr = open(xmlfilename).read()
            elif zipfilename is not None:
                zf = zipfile.ZipFile(zipfilename, 'r')
                filenames = [zi.filename for zi in zf.infolist()]
                safeDirName = [fn for fn in filenames if fn.endswith('.SAFE/')][0]
                bn = safeDirName.replace('.SAFE/', '')
                # The meta filename is, rather ridiculously, named something slightly different 
                # inside the SAFE directory, so we have to construct that name. 
                metafilename = bn.replace('PRD', 'MTD').replace('MSIL1C', 'SAFL1C') + ".xml"
                fullmetafilename = safeDirName + metafilename
                if fullmetafilename not in filenames:
                    # We have a new format package, in which the meta filename is constant. 
                    fullmetafilename = safeDirName + 'MTD_MSIL1C.xml'
                if fullmetafilename not in filenames:
                    # We have a new format package, in which the meta filename is constant. 
                    fullmetafilename = safeDirName + 'MTD_MSIL2A.xml'
                mf = zf.open(fullmetafilename)
                xmlStr = mf.read()
                del mf
                
                # Read in the raw content of the preview image png file, and stash on the object
                previewFilename = bn.replace('PRD', 'BWI') + ".png"
                previewFullFilename = safeDirName + previewFilename
                if previewFullFilename not in filenames:
                    # Perhaps we have a new format package, with the preview image as 
                    # a jp2 in the QI_DATA directory
                    previewFullFilenameList = [fn for fn in filenames 
                        if fnmatch.fnmatch(fn, '*/GRANULE/*/QI_DATA/*PVI.jp2')]
                    if len(previewFullFilenameList) > 0:
                        previewFullFilename = previewFullFilenameList[0]
                if previewFullFilename in filenames:
                    try:
                        pf = zf.open(previewFullFilename)
                        self.previewImgBin = pf.read()
                        del pf
                    except zipfile.BadZipfile:
                        pass
                
                # Read in the whole set of tile-level XML files, too, so we can 
                # grab tileId values from them
                self.tileNameList = None
                # This is currently commented out, as it adds significant run-time. I
                # expect to return to this in future. 
#                tileXmlPattern = safeDirName + "GRANULE/*/*.xml"
#                tileXmlFiles = [fn for fn in filenames if fnmatch.fnmatch(fn, tileXmlPattern)]
#                tileIdSet = set()
#                for tileXml in tileXmlFiles:
#                    tf = zf.open(tileXml)
#                    tileMeta = Sen2TileMeta(fileobj=tf)
#                    del tf
#                    tileIdSet.add(tileMeta.tileId[1:].upper())
#                self.tileNameList = sorted(list(tileIdSet))
        
        self.zipfileMetaXML = xmlStr
        
        doc = minidom.parseString(xmlStr)
        
        generalInfoNode = doc.getElementsByTagName('n1:General_Info')[0]
        geomInfoNode = doc.getElementsByTagName('n1:Geometric_Info')[0]
        auxilInfoNode = doc.getElementsByTagName('n1:Auxiliary_Data_Info')[0]
        qualInfoNode = doc.getElementsByTagName('n1:Quality_Indicators_Info')[0]
        
        self.processingLevel = findElementByXPath(generalInfoNode, 'Product_Info/PROCESSING_LEVEL')[0].firstChild.data.strip()
        self.spacecraftName = findElementByXPath(generalInfoNode, 'Product_Info/Datatake/SPACECRAFT_NAME')[0].firstChild.data.strip()
        self.satId = "S" + self.spacecraftName.split('-')[1]
        self.processingSoftwareVersion = findElementByXPath(generalInfoNode, 
            'Product_Info/PROCESSING_BASELINE')[0].firstChild.data.strip()
        
        # The image acquisition start and stop times. In older versions of the ESA processing
        # software, note that start and stop times were identical (which is obviously wrong)
        prodStartTimeNode = findElementByXPath(generalInfoNode, 'Product_Info/PRODUCT_START_TIME')[0]
        prodStartTimeStr = prodStartTimeNode.firstChild.data.strip()
        self.startTime = datetime.datetime.strptime(prodStartTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        prodStopTimeNode = findElementByXPath(generalInfoNode, 'Product_Info/PRODUCT_STOP_TIME')[0]
        prodStopTimeStr = prodStopTimeNode.firstChild.data.strip()
        self.stopTime = datetime.datetime.strptime(prodStopTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        # Product generation time, i.e. when ESA processed it
        generationTimeNode = findElementByXPath(generalInfoNode, 'Product_Info/GENERATION_TIME')[0]
        generationTimeStr = generationTimeNode.firstChild.data.strip()
        self.generationTime = datetime.datetime.strptime(generationTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        relOrbitStr = findElementByXPath(generalInfoNode, 'Product_Info/SENSING_ORBIT_NUMBER')[0].firstChild.data.strip()
        self.relativeOrbitNumber = int(relOrbitStr)
        
        # The cloud indicator
        cloudPcntNode = findElementByXPath(qualInfoNode, 'Cloud_Coverage_Assessment')[0]
        self.cloudPcnt = float(cloudPcntNode.firstChild.data.strip())
        
        # The full extPos footprint. This is a very poor excuse for a footprint, but it will
        # do for now. 
        extPosNode = findElementByXPath(geomInfoNode, 'Product_Footprint/Product_Footprint/Global_Footprint/EXT_POS_LIST')[0]
        coordsList = [float(v) for v in extPosNode.firstChild.data.strip().split()]
        x = coordsList[1::2]
        y = coordsList[0::2]
        coords = [[x, y] for (x, y) in zip(x, y)]

        footprintGeom = geomutils.geomFromOutlineCoords(coords)
        prefEpsg = geomutils.findSensibleProjection(footprintGeom)
        self.centroidXY = geomutils.findCentroid(footprintGeom, prefEpsg)
        self.extPosWKT = footprintGeom.ExportToWkt()

        # Special values in imagery
        scaleValNodeList = findElementByXPath(generalInfoNode, 'Product_Image_Characteristics/QUANTIFICATION_VALUE')
        if len(scaleValNodeList) > 0:
            scaleValNode = scaleValNodeList[0]
            self.scaleValue = float(scaleValNode.firstChild.data.strip())
        else:
            # We might be in a L2A file, in which case there are several scale values for different products
            scaleValNodeList = findElementByXPath(generalInfoNode, 'Product_Image_Characteristics/QUANTIFICATION_VALUES_LIST')
            if len(scaleValNodeList) > 0:
                scaleValNode = scaleValNodeList[0]
                refScaleNode = findElementByXPath(scaleValNode, 'BOA_QUANTIFICATION_VALUE')[0]
                self.scaleValue = float(refScaleNode.firstChild.data.strip())
                aotScaleNode = findElementByXPath(scaleValNode, 'AOT_QUANTIFICATION_VALUE')[0]
                self.aotScaleValue = float(aotScaleNode.firstChild.data.strip())
                wvpScaleNode = findElementByXPath(scaleValNode, 'WVP_QUANTIFICATION_VALUE')[0]
                self.wvpScaleValue = float(wvpScaleNode.firstChild.data.strip())
        specialValuesNodeList = findElementByXPath(generalInfoNode, 'Product_Image_Characteristics/Special_Values')
        # These guys have no idea how to use XML properly. Sigh......
        for node in specialValuesNodeList:
            name = node.getElementsByTagName('SPECIAL_VALUE_TEXT')[0].firstChild.data.strip()
            value = node.getElementsByTagName('SPECIAL_VALUE_INDEX')[0].firstChild.data.strip()
            if name == "NODATA":
                self.nullVal = int(value)
            elif name == "SATURATED":
                self.saturatedVal = int(value)


def findElementByXPath(node, xpath):
    """
    Find a sub-node under the given XML minidom node, by traversing the given XPATH notation. 
    Searches all possible values, and returns a list of whatever it finds which matches. 
    
    It would be better if minidom understood XPATH, but it does not seem to. I could use
    some of the other libraries, but ElementTree seems to have obscure bugs, and I did 
    not want to introduce any other dependencies. Sigh.....
    
    """
    nodeNameList = xpath.split('/')
    if len(nodeNameList) > 1:
        nextNodeList = node.getElementsByTagName(nodeNameList[0])
        nodeList = []
        for n in nextNodeList:
            nodeList.extend(findElementByXPath(n, '/'.join(nodeNameList[1:])))
    else:
        nodeList = node.getElementsByTagName(nodeNameList[0])

    return nodeList


def calcMGRSnameFromCoords(epsg, easting, northing):
    """
    Calculate the US-MGRS 100km grid square identifier, which is then used for the 
    the ESA tile ID string. This should not be necessary, as ESA have supplied the
    tile name in the granule names, but they appear to have got it wrong sometimes. 
    So, in order to make sure we have it right, I calculate it myself. How very
    tiresome. 
    
    The inputs are the easting and northing in UTM coords, along with the EPSG
    number of the relevant UTM projection. It is assumed that the EPSG is for
    the WGS84 spheroid/datum, and hence is either 326nn or 327nn (north or 
    south hemisphere). 

    My reference on how to do this is mostly 
        https://en.wikipedia.org/wiki/Military_grid_reference_system
    but also with help from 
        http://earth-info.nga.mil/GandG/publications/tm8358.1/tr83581f.html
    and some source code at
        http://earth-info.nga.mil/GandG/geotrans/index.html 
    From the source code page, download mgrs.tgz, and look in MGRS.cpp. The
    routines MGRS::getGridValues() and MGRS::fromUTM() are particularly instructive. 

    I have not coded the variations for the polar regions, nor for Svalbard and 
    south-west Norway. However, it appears that ESA have not used the special polar 
    cases anyway. For latitude < -80 or > +84, MGRS is supposed to use Universal
    Polar Stereographic projection (UPS) instead of UTM, but it seems that the 
    ESA processing software leaves everything in UTM, and assigns a tile name
    according to the UTM algorithm anyway. From my testing, it seems this routine
    gets the same tile name as ESA, so I am leaving it that way. 
    
    I have no idea what ESA do in Svarlbard or south-west Norway. 

    """
    TWOMILLION = 2000000
    ONEHUNDREDTHOUSAND = 100000
    FIVEHUNDREDTHOUSAND = 500000

    # Find lat/long for easting/northing
    srUTM = osr.SpatialReference()
    srUTM.ImportFromEPSG(int(epsg))
    srLL = osr.SpatialReference()
    srLL.ImportFromEPSG(4326)
    tr = osr.CoordinateTransformation(srUTM, srLL)
    (longitude, latitude, z) = tr.TransformPoint(easting, northing)
    # We need these to do the latitude band in a way that matches ESA's approach.
    # Note that we do not actually know what ESA's approach is, we are 
    # just reverse-engineering it to match. 
    (longitudeWestZoneEdge, latitudeWestZoneEdge, z) = tr.TransformPoint(200000, northing)
    (longitudeEastZoneEdge, latitudeEastZoneEdge, z) = tr.TransformPoint(700000, northing)
    avgLatitude = (latitudeWestZoneEdge + latitudeEastZoneEdge) / 2.0

    # Get UTM zone, from EPSG
    epsgStr = str(epsg)
    if not (epsgStr.startswith("327") or epsgStr.startswith("326")):
        raise Sen2MetaError("Cannot determine UTM zone from EPSG:{}".format(epsgStr))
    utmZone = int(epsgStr[3:])
    # Technically, the US-MGRS tile specification has special cases for the poles, but
    # it seems that ESA do not use them. However, if they do, this next trap would
    # at least raise an exception and thus prevent us from doing the wrong thing. 
    if utmZone == 61:
        raise Sen2MetaError("Projection is Universal Polar Stereographic, which we do not yet cope with")

    utmZoneStr = "{:02}".format(utmZone)

    # Note that these do NOT use 'I' or 'O', hence I have to remove them explicitly
    iA = ord('A')
    iZ = ord('Z')
    letters = [chr(i) for i in range(iA, iZ+1)]
    letters.remove('I')
    letters.remove('O')

    latitudebandStart = -80
    latitudebandWidth = 8
    # Note that we use the average latitude across this level in the UTM zone. This is
    # so we can match how ESA do it (without actually knowing how ESA do it), especially in
    # the tiles on the boundary between J and K latitude bands, in the centre of each zone. 
    latbandNdx = int((avgLatitude - latitudebandStart) / latitudebandWidth) + letters.index('C')
    # Only allow latitude band letters up to X. Should do something to 
    # cope with polar regions, which are supposed to be different......
    latbandNdx = min(letters.index('X'), latbandNdx)
    latbandLetter = letters[latbandNdx]

    # Column letter, which is based on which zone we are in, and which multiple of
    # 100000 the easting is
    zoneGroup = (utmZone % 6)
    if zoneGroup == 0:
        zoneGroup = 6
    # The left-most column letter in the grid zone
    if zoneGroup in (1, 4):
        colLetterNdx = letters.index('A')
    elif zoneGroup in (2, 5):
        colLetterNdx = letters.index('J')
    elif zoneGroup in (3, 6):
        colLetterNdx = letters.index('S')
    # Offset to the column letter for this easting
    colLetterNdx += int(easting / ONEHUNDREDTHOUSAND) - 1
    colLetter = letters[colLetterNdx]

    # The row letter, which is based on the odd/even-ness of the zone, and which 
    # multiple of 100000 the northing is in
    northingOffset = 0
    if utmZone % 2 == 0:
        northingOffset = FIVEHUNDREDTHOUSAND

    nrthWithOffset = northing + northingOffset
    nrthRemainder = nrthWithOffset - int(nrthWithOffset / TWOMILLION) * TWOMILLION
    rowLetterNdx = int(nrthRemainder / ONEHUNDREDTHOUSAND)
    rowLetter = letters[rowLetterNdx]

    mgrsString = utmZoneStr + latbandLetter + colLetter + rowLetter
    return mgrsString


class Sen2MetaError(Exception): pass
