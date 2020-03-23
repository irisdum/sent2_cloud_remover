"""
Functions for interface to the SARA client API. 

These routines can be used to build client-side applications for searching and 
downloading data. 

The most obvious way to use these routines is as follows::

    urlOpener = saraclient.makeUrlOpener()
    sentinel = 2
    paramList = ['startDate=2017-05-01', 'completionDate=2017-05-31']
    results = saraclient.searchSara(urlOpener, sentinel, paramList)

This would return a list of multi-level dictionary objects created from the JSON output 
of the server, one for each matching zipfile. The paramList can be any of the parameters which
the SARA API accepts, these are passed straight through to the API. 

The default SARA server is hard-wired in this module. However, the server name, and the protocol
to be used, can both be over-ridden using the following environment variables
    | AUSCOPHUB_SARA_PROTOCOL (default https)
    | AUSCOPHUB_SARA_SERVERHOST (default copernicus.nci.org.au)

"""
from __future__ import print_function, division

import sys
import os
import json
import copy
import shlex
import subprocess


from urllib.request import build_opener, ProxyHandler
from urllib.error import HTTPError
from urllib.parse import quote as urlquote


SARA_PROTOCOL = os.getenv("AUSCOPHUB_SARA_PROTOCOL", default="https")
SARA_HOST = os.getenv("AUSCOPHUB_SARA_SERVERHOST", default="copernicus.nci.org.au")

SARA_SEARCHSERVER = "{}://{}/sara.server/1.0/api/collections".format(SARA_PROTOCOL, SARA_HOST)


def makeUrlOpener(proxy=None):
    """
    Use the crazy urllib2 routines to make a thing which can open a URL, with proxy
    handling if required. Return an opener object, which is used as::
        reader = opener.open(url)
        
    """
    if proxy is None:
        opener = build_opener()
    else:
        proxyHandler = ProxyHandler({'http':proxy, 'https':proxy})
        opener = build_opener(proxyHandler)
    return opener


def searchSara(urlOpener, sentinelNumber, paramList):
    """
    Search the GA/NCI SARA Resto API, according to a set of parameter
    name/value pairs, as given in paramList. The names and values are those
    allowed by the API, as described at
        | http://copernicus.nci.org.au/sara.server/1.0/api/collections/describe.xml
        | http://copernicus.nci.org.au/sara.server/1.0/api/collections/S1/describe.xml
        | http://copernicus.nci.org.au/sara.server/1.0/api/collections/S2/describe.xml
        | http://copernicus.nci.org.au/sara.server/1.0/api/collections/S3/describe.xml
    Each name/value pair is added to a HTTP GET URL as a separate name=value 
    string, separated by '&', creating a single query. 
    
    The overall effect of multiple name/value pairs is that each one further 
    restricts the results, in other words they are being AND-ed together. Note 
    that this is not because of the '&' in the constructed URL, that is just the 
    URL separator character. This means that there is no mechanism for doing an 
    OR of multiple search conditions. 
    
    If sentinelNumber is None, then all Sentinels are searched, using the "all collections"
    URL of the API. I am not sure how useful that might be. 
    
    Args:
        urlOpener:  Object as created by the makeUrlOpener() function
        sentinelNumber (int): an integer (i.e. 1, 2 or 3), identifying which Sentinel family
        paramList (list): List of name=value strings, corresponding to the query parameters
                defined by the SARA API. 
    
    Returns:
        The return value is a list of the matching datasets. Each entry is a feature object, 
        as given by the JSON output of the SARA API. This list is built up from multiple
        queries, because the server pages its output, so the list is just the feature objects,
        without all the stuff which would be repeated per page. 
    
    """
    url = makeQueryUrl(sentinelNumber, paramList)
    print(url)

    (results, httpErrorStr) = readJsonUrl(urlOpener, url)
    #print(results, httpErrorStr)
    if httpErrorStr is not None:
        print("Error querying URL:", url, file=sys.stderr)
        raise SaraClientError(httpErrorStr)
    
    # Start with the first page of results. 
    allFeatures = results['features']
    
    # The API only gives us a page of results at a time. So, we have to do repeated queries,
    # with increasing page numbers, to get all pages. We can't use the totalResults field to work
    # out how many pages there ought to be, because Resto does something crazy with that, 
    # so instead we have to just keep going until we don't get any more results. All a bit
    # unsatisfactory, but this is what we have. 
    finished = False
    page = 2
    while not finished:
        tmpParamList = copy.copy(paramList)
        tmpParamList.append('page={}'.format(page))
        url = makeQueryUrl(sentinelNumber, tmpParamList)
        (results, httpErrorStr) = readJsonUrl(urlOpener, url)
        features = results['features']

        if len(features) > 0:
            allFeatures.extend(features)
            page += 1
        else:
            finished = True
    
    return allFeatures


def makeQueryUrl(sentinelNumber, paramList):
    """
    Return a full URL for the query defined by the given parameters
    """
    # No URL encoding for these characters
    noURLencode = "=:/(),"
    queryStr = '&'.join([urlquote(p, safe=noURLencode) for p in paramList])

    if sentinelNumber is None:
        url = "{}/search.json?{}".format(SARA_SEARCHSERVER, queryStr)
    else:
        url = "{}/S{}/search.json?{}".format(SARA_SEARCHSERVER, sentinelNumber, queryStr)

    return url


def readJsonUrl(urlOpener, url):
    """
    Read the contents of the given URL, returning the object created from the
    JSON which the server returns
    """
    try:
        reader = urlOpener.open(url)
        jsonStr = reader.read()
        # Ensure that we have a str object, but in a robust way. Mainly required in Python-3. 
        if hasattr(jsonStr, 'decode'):
            jsonStr = jsonStr.decode('utf-8')
        results = json.loads(jsonStr)
        httpErrorStr = None
    except HTTPError as e:
        results = None
        httpErrorStr = str(e)
    return (results, httpErrorStr)


def simplifyFullFeature(feature):
    """
    Given a full feature object as returned by the server (a GeoJSON-compliant object),
    extract just the few interesting pieces, and return a single dictionary of them. 
    The names are ones I made up, and do not comply with any particular standards or anything. 
    They are intended purely for internal use within this software. 
    
    """
    d = {}
    
    for localName in [FEATUREATTR_DOWNLOADURL, FEATUREATTR_MD5, FEATUREATTR_SIZE, 
            FEATUREATTR_ESAID, FEATUREATTR_CLOUDCOVER]:
        d[localName] = getFeatAttr(feature, localName)
    
    return d


FEATUREATTR_DOWNLOADURL = "downloadurl"
FEATUREATTR_MD5 = "md5"
FEATUREATTR_SIZE = "size"
FEATUREATTR_ESAID = "esaid"
FEATUREATTR_CLOUDCOVER = "cloud"

def getFeatAttr(feature, localName):
    """
    Given a feature dictionary as returned by the SARA API, and the local name for some 
    attribute of interest, this function knows how to navigate through the feature 
    structures to find the relevant attribute. 
    
    The main reason for this function is to give a simple, flat namespace, without requiring 
    that other parts of the code decompose the multi-level structure of the feature objects. 
    
    Note that the local names are NOT the same as the names in the feature structure, but 
    are simple local names used unambiguously. Only a subset of attributes are handled. 
    
    """
    value = None
    
    properties = feature['properties']
    download = properties['services']['download']
    if localName == FEATUREATTR_DOWNLOADURL:
        value = download['url']
    elif localName == FEATUREATTR_MD5:
        checksum = download['checksum']
        checksumParts = checksum.split(':')
        if checksumParts[0] == "md5":
            value = checksumParts[1]
    elif localName == FEATUREATTR_SIZE:
        value = download['size']
    elif localName == FEATUREATTR_ESAID:
        value = properties['productIdentifier']
    elif localName == FEATUREATTR_CLOUDCOVER:
        value = properties['cloudCover']

    return value


def getRemoteFilename(downloadUrl, proxy):
    """
    Given the SARA download URL, ask the server what the actual filename is. 
    
    At the moment, this uses "curl -I" to do the work, but I would much prefer to do
    this directly in Python. In theory this should be possible, but I can't get 
    the authentication to work. When I do, I will change this code, and thus may 
    require extra arguments. I also suspect that the re-directs which the SARA server
    does will cause me trouble, but I have yet to get to that point. 
    
    In the meantime, this is slow, but at least it works. 
    
    I am a bit unsure about this approach......
    
    """
    cmdWords = ["curl", "--silent", "-n", "-L", "-I", downloadUrl]
    if proxy is not None:
        cmdWords.extend(["-x", proxy])
    
    proc = subprocess.Popen(cmdWords, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    (stdout, stderr) = proc.communicate()
    
    # I should really parse this with the standard library tools for doing so. However, 
    # because of the redirections the server does, this is actually several sets of HTTP headers
    # kind of tacked together, which means there are extra traps. So, given that I have
    # to handle at least part of it myself, I decided to just handle the whole lot. 
    stdoutLines = stdout.strip().split('\n')
    filename = None
    for line in stdoutLines:
        if line.startswith('Content-Disposition: '):
            words = shlex.split(line)
            for word in words:
                if word.startswith("filename="):
                    fields = word.split('=')
                    filename = fields[1]
    return filename


class SaraClientError(Exception): pass
