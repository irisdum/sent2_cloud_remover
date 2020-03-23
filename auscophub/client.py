"""
Functions for client programs to access to the AusCopernicusHub server. 
Initially this is centred around accessing the THREDDS server, but other client
access methods could go in here as well. 

These routines can be used to build client-side applications for searching and 
downloading data. 

"""
from __future__ import print_function, division

import sys
from xml.dom import minidom
import xml.parsers.expat

from auscophub import auscophubmeta


isPython3 = (sys.version_info.major == 3)
if isPython3:
    from urllib.request import build_opener, ProxyHandler
else:
    from urllib2 import build_opener, ProxyHandler

THREDDS_BASE = "http://dapds00.nci.org.au/thredds"
THREDDS_CATALOG_BASE = "{}/catalog".format(THREDDS_BASE)
THREDDS_FILES_BASE = "{}/fileServer".format(THREDDS_BASE)
THREDDS_COPERNICUS_SUBDIR = "fj7/Copernicus"
THREDDS_SEN1_CATALOG_BASE = "{}/{}/Sentinel-1".format(THREDDS_CATALOG_BASE, THREDDS_COPERNICUS_SUBDIR)
THREDDS_SEN2_CATALOG_BASE = "{}/{}/Sentinel-2".format(THREDDS_CATALOG_BASE, THREDDS_COPERNICUS_SUBDIR)
THREDDS_SEN3_CATALOG_BASE = "{}/{}/Sentinel-3".format(THREDDS_CATALOG_BASE, THREDDS_COPERNICUS_SUBDIR)


def makeUrlOpener(proxy=None):
    """
    Use the crazy urllib2 routines to make a thing which can open a URL, with proxy
    handling if required. Return an opener object, which is used as
        reader = opener.open(url)
        
    """
    if proxy is None:
        opener = build_opener()
    else:
        proxyHandler = ProxyHandler({'http':proxy})
        opener = build_opener(proxyHandler)
    return opener


def getDescriptionMetaFromThreddsByBounds(urlOpener, sentinelNumber, instrumentStr, 
        productId, startDate, endDate, longLatBoundingBox):
    """
    Search the THREDDS server and return a list of AusCopHubMeta objects for
    the given sentinel number, the given product, and are within the time and
    location bounds given. These can be filtered further.
    
    Args:
        urlOpener: Object as created by the makeUrlOpener() function
        sentinelNumber (int): an integer (i.e. 1, 2 or 3), identifying which Sentinel family
        instrumentStr (str): Which instrument - specific to each Sentinel family. Possible values
            are - Sentinel-1: C-SAR; Sentinel-2: MSI; Sentinel-3: one of {OLCI, SLSTR, SRAL, MWR}.
        productId (str): Which data product - specific to each Sentinel family. Possible values
            are - Sentinel-1: one of {SLC, GRD}; 
            Sentinel-2: L1C;
            Sentinel-3: As yet unknown. 
        startdate (str): Earliest acquisition date to include, as yyyymmdd
        endDate (str): Latest acquisition date to include, as yyyymmdd
        boundingBox (tuple): Search region lat/long bounding box in decimal degrees, in
                the form (westLong, eastLong, southLat, northLat), with negative values for
                south of equator and west of Greenwich. 

    In future, support may be added for products RAW and OCN for Sentinel-1, and L2A for Sentinel-2. 

    Warning:
        Note that we do not (yet) cope with the boundingBox crossing the international 
        date line
    
    Returns:
        A list of tuples of the form (urlStr, metaObj)
        where urlStr is the URL of the XML file on the server, and metaObj is 
        the AusCopHubMeta object holding the contents of the XML file, read from the 
        server.
    
    """
    # This assumes we can just use the instrumentStr and productId string directly in the URL. 
    # This may not always be true, but see how we go. 
    if sentinelNumber == 1:
        productCatalogUrl = "{}/{}/{}".format(THREDDS_SEN1_CATALOG_BASE, instrumentStr, productId)
    elif sentinelNumber == 2:
        productCatalogUrl = "{}/{}/{}".format(THREDDS_SEN2_CATALOG_BASE, instrumentStr, productId)
    elif sentinelNumber == 3:
        productCatalogUrl = "{}/{}/{}".format(THREDDS_SEN3_CATALOG_BASE, instrumentStr, productId)
    else:
        raise AusCopHubClientError("Unknown sentinel number {}".format(sentinelNumber))
    
    # Find the top-level year directories
    yearLists = ThreddsServerDirList(urlOpener, productCatalogUrl)
    if len(yearLists.subdirs) == 0:
        raise AusCopHubClientError("Cannot find year directories. Check the server '{}'".format(productCatalogUrl))
        
    startYMwithDash = "{}-{}".format(startDate[:4], startDate[4:6])
    endYMwithDash = "{}-{}".format(endDate[:4], endDate[4:6])
    startYMDwithDash = "{}-{}-{}".format(startDate[:4], startDate[4:6], startDate[6:])
    endYMDwithDash = "{}-{}-{}".format(endDate[:4], endDate[4:6], endDate[6:])
    
    # Create a list of catalog objects for yyyy-mm subdirs which are in the date range
    ymCatalogObjList = []
    for subdirObj in yearLists.subdirs:
        ymLists = ThreddsServerDirList(urlOpener, subdirObj.fullUrl)
        for ymSubdirObj in ymLists.subdirs:
            yearMonthWithDash = ymSubdirObj.title
            if yearMonthWithDash >= startYMwithDash and yearMonthWithDash <= endYMwithDash:
                ymCatalogObjList.append(ymSubdirObj)
    
    # Sentinel-3 is divided into individual days, but Sentinels 1 and 2 are divided to whole month.
    # So, here, we create a list of catalog objects which represent the finest temporal
    # division available, in either case, and this is used in the subsequent steps. 
    if sentinelNumber == 3:
        dayCatalogObjList = []
        for ymSubdirObj in ymCatalogObjList:
            dayLists = ThreddsServerDirList(urlOpener, ymSubdirObj.fullUrl)
            for ymdSubdirObj in dayLists.subdirs:
                ymdWithDash = ymdSubdirObj.title
                if ymdWithDash >= startYMDwithDash and ymdWithDash <= endYMDwithDash:
                    dayCatalogObjList.append(ymdSubdirObj)
        timeCatalogObjList = dayCatalogObjList
    elif sentinelNumber in [1, 2]:
        timeCatalogObjList = ymCatalogObjList
    
    # Create a list of catalog objects for grid cell subdirs which intersect the bounding box. 
    gridCellCatalogObjList = []
    (westLong, eastLong, southLat, northLat) = longLatBoundingBox
    for ymSubdirObj in timeCatalogObjList:
        cellLists = ThreddsServerDirList(urlOpener, ymSubdirObj.fullUrl)
        for cellDirObj in cellLists.subdirs:
            gridCellDirName = cellDirObj.title
            if gridCellDirWithinBounds(gridCellDirName, northLat, southLat, westLong, eastLong):
                gridCellCatalogObjList.append(cellDirObj)
    
    if len(gridCellCatalogObjList) == 0:
        # Perhaps this is one of the products which has no spatial division, so just look in 
        # lowest temporal division directory
        gridCellCatalogObjList = timeCatalogObjList
    
    # Create a list of dataset objects for every XML file in the given list of catalog objects. 
    dsObjList = []
    for subdirObj in gridCellCatalogObjList:
        dirlists = ThreddsServerDirList(urlOpener, subdirObj.fullUrl)
        dsObjList.extend([dsObj for dsObj in dirlists.datasets if dsObj.name.endswith(".xml")])
    
    # Create a list of the meta files and their contents
    metaList = []
    for dsObj in dsObjList:
        url = dsObj.fullUrl
        xmlStr = urlOpener.open(url).read()
        try:
            metaObj = auscophubmeta.AusCopHubMeta(xmlStr=xmlStr)
        except Exception:
            metaObj = None
        
        if metaObj is not None:
            # Filter by exact date, instead of just month, as above
            yyyymmdd = metaObj.startTime.strftime("%Y%m%d")
            if yyyymmdd >= startDate and yyyymmdd <= endDate:
                metaList.append((url, metaObj))
    
    return metaList


def gridCellDirWithinBounds(gridCellDirName, northLat, southLat, westLong, eastLong):
    """
    Return True if the given grid cell directory name lies at least partially within the 
    lat/long bounds given. 
        
    """
    # Decode the grid cell bounds from the string. Assumes a fixed format. 
    gcNorthLat = decodeDegreesStr(gridCellDirName[:3])
    gcWestLong = decodeDegreesStr(gridCellDirName[3:7])
    gcSouthLat = decodeDegreesStr(gridCellDirName[8:11])
    gcEastLong = decodeDegreesStr(gridCellDirName[11:15])
    
    withinEastWest = ((gcEastLong >= westLong) and (gcWestLong <= eastLong))
    withinNorthSouth = ((gcSouthLat <= northLat) and (gcNorthLat >= southLat))
    within = withinEastWest and withinNorthSouth
    return within


def decodeDegreesStr(valStr):
    """
    Return a signed latitude/longitude value from a string. Only copes with the integer 
    values used in grid cell names. 
    """
    val = int(valStr[:-1])
    if valStr[-1] in ("S", "W"):
        val = -val
    return val

def loadDatasetDescriptionXmlList(urlOpener, xmlDatasetEntryList):
    """
    Given a list of DatasetEntry object, all of which correspond to .xml files
    on the server, load the XML into AusCopHubMeta objects and return 
    a list of these objects. 
    
    """
    metaObjList = []
    for dsObj in xmlDatasetEntryList:
        if not dsObj.name.endswith('.xml'):
            raise AusCopHubClientError("DatasetEntry object '{}' does not end in '.xml'".format(dsObj.name))

        xmlStr = urlOpener.open(dsObj.fullUrl).read()
        metaObj = auscophubmeta.AusCopHubMeta(xmlStr=xmlStr)
        metaObjList.append(metaObj)
    return metaObjList


class ThreddsServerDirList(object):
    """
    Connect to the THREDDS server and create an object which lists the
    "interesting" pieces of the catalog.xml for the given subdirectory.
    
    Attributes:
        subdirs: list of ThreddsCatalogRefEntry objects, for subdirectories under this one, i.e. <catalogRef> tags
        datasets: list of ThreddsDatasetEntry objects, for datasets in this subdir, i.e. <dataset> tags
        
    """
    def __init__(self, urlOpener, subdirUrl):
        doc = getThreddsCatalogXml(urlOpener, subdirUrl)        
        
        catalogNode = doc.getElementsByTagName('catalog')[0]
        topDatasetNode = catalogNode.getElementsByTagName('dataset')[0]
        
        subdirNodeList = topDatasetNode.getElementsByTagName('catalogRef')
        self.subdirs = [ThreddsCatalogRefEntry(node, subdirUrl) for node in subdirNodeList]
        
        datasetNodeList = topDatasetNode.getElementsByTagName('dataset')
        self.datasets = [ThreddsDatasetEntry(node) for node in datasetNodeList]


class ThreddsDatasetEntry(object):
    """
    Details of a <dataset> tag in the catalog.xml
    """
    def __init__(self, datasetNode):
        self.name = datasetNode.getAttribute('name').strip()
        self.urlPath = datasetNode.getAttribute('urlPath').strip()
        self.fullUrl = "{}/{}".format(THREDDS_FILES_BASE, self.urlPath)


class ThreddsCatalogRefEntry(object):
    """
    Details of a <catalogRef> tag in the catalog.xml
    """
    def __init__(self, catalogRefNode, baseUrl):
        self.href = catalogRefNode.getAttribute('xlink:href').strip()
        self.idStr = catalogRefNode.getAttribute('ID').strip()
        self.title = catalogRefNode.getAttribute('xlink:title').strip()
        self.fullUrl = "{}/{}".format(baseUrl.strip("/catalog.xml"), self.href)


def getThreddsCatalogXml(urlOpener, baseUrl, returnXmlString=False):
    """
    Get the catalog.xml file for the given baseUrl
    By default, it will parse the catalog.xml using minidom and return the document object. 
    If the XML fails to parse, then the return value will be None.
    
    If returnXmlString is True, then just return the XML string, without parsing. 
    
    """
    url = baseUrl
    if not url.endswith("/catalog.xml"):
        url = "{}/catalog.xml".format(baseUrl)
    reader = urlOpener.open(url)
    xmlStr = reader.read()
    
    if returnXmlString:
        returnVal = xmlStr
    else:
        try:
            returnVal = minidom.parseString(xmlStr)
        except xml.parsers.expat.ExpatError:
            returnVal = None
        
    return returnVal


class AusCopHubClientError(Exception): pass
