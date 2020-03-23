"""
Some utility functions for handling the geometries given with Sentinel files. 

ESA have done a fairly minimal job with their footprint geometries. The routines in this
file are mostly aimed at providing a bit more support beyond that, mostly for things like
crossing the International Date Line. 

It is assumed that we have access to the GDAL/OGR library, and numpy. 

Note on efficiency. 
------------------
These routines all operate on the basic ogr.Geometry object. However, since this is 
a relatively opaque structure, mostly they will convert to JSON strings and
use those to get hold of the coordinates. This means there is a fair bit of
converting back and forth, which means we are not at maximum efficiency. There is also
quite a bit of changing of projection. However, in the context of AusCopHub, this is 
all somewhat negligible compared to doing things like reading and copying ESA zip files. 

"""
from __future__ import print_function, division

import numpy
from osgeo import ogr, osr


def findSensibleProjection(geom):
    """
    Given a polygon Geometry object in lat/long, work out what would be a suitable projection
    to use with this area, in order to avoid things like the international 
    date line wrap-around, or the north/sourth pole discontinuities. 
    
    This only makes sense for tiled products, as opposed to long strips which cross 
    multiple zones, etc. 
    
    Main possible options are UTM in a suitable zone, UPS when near the poles. 
    
    Return the EPSG number of the projection. 
    
    """
    coords = getCoords(geom)
    y = coords[:, 1]
    x = coords[:, 0]
    yMin = y.min()
    yMax = y.max()
    if (yMax - yMin) > 90:
        # We are crossing a lot of latitude, which suggests that we have a 
        # long strip> In this case, we don't even bother to suggest an EPSG. 
        epsg = None
    elif yMin < -80:
        # We are nearing the south pole, so go with UPS south
        epsg = 32761
    elif yMax > 80:
        # Nearing north pole, so UPS North
        epsg = 32661
    else:
        # Work out a UTM zone. Note that we use the median value to get a rough 
        # idea of the centre, rather than the mean, because the mean is subject to all 
        # sorts of problems when crossing the date line
        xMedian = numpy.median(x)
        yMedian = numpy.median(y)
        zone = int((xMedian + 180)/6) % 60 + 1
        if yMedian < 0:
            epsgBase = 32700
        else:
            epsgBase = 32600
        epsg = epsgBase + zone
    return epsg


def makeTransformations(epsg1, epsg2):
    """
    Make a pair of ogr.CoordinateTransformation objects, for transforming between
    the two given EPSG projections.
    
    Return a tuple
        (tr1to2, tr2to1)
    
    """
    sr1 = osr.SpatialReference()
    sr1.ImportFromEPSG(epsg1)
    sr2 = osr.SpatialReference()
    sr2.ImportFromEPSG(epsg2)
    tr1to2 = osr.CoordinateTransformation(sr1, sr2)
    tr2to1 = osr.CoordinateTransformation(sr2, sr1)
    return (tr1to2, tr2to1)


def findCentroid(geom, preferredEpsg):
    """
    Given a geometry as a lat/long polygon, find the lat/long centroid, by first projecting
    into the preferred EPSG, so as to avoid discontinuities. The preferredEpsg is one
    in which the polygon ought to make sense (as found, hopefully, 
    by the findSensibleProjection() function). 
    
    Returns a pair [centroidX, centroidY] in lat/long
    
    """
    (projTr, llTr) = makeTransformations(4326, preferredEpsg)
    
    geomProj = copyGeom(geom)
    geomProj.Transform(projTr)
    geomCentroid = geomProj.Centroid()
    geomCentroid.Transform(llTr)
    
    centroidDict = eval(geomCentroid.ExportToJson())
    centroidXY = centroidDict['coordinates']
    return centroidXY


def copyGeom(geom):
    """
    Return a copy of the geometry. OGR does not provide a good method for doing this. 
    """
    geomJson = geom.ExportToJson()
    newGeom = ogr.CreateGeometryFromJson(geomJson)
    return newGeom


def getCoords(geom):
    """
    Return the coordinates of the given OGR geometry. Assumes that this is a single 
    polygon, and returns a numpy array of the x, y coords, of shape (numPts, 2).
    
    If the polygon has holes, they will be discarded - this is just the outer polygon. 
    
    If the geometry is a MultiPoint geom, also return a 2-d array of coords. 
    
    """
    geomDict = eval(geom.ExportToJson())
    coords = geomDict['coordinates']
    if geomDict['type'] == 'Polygon':
        coordsArray = numpy.array(coords[0])
    elif geomDict['type'] == 'MultiPoint':
        coordsArray = numpy.array(coords)
    else:
        coordsArray = None
    return coordsArray


def geomFromOutlineCoords(coords):
    """
    The given list of pairs (or 2-d numpy array) is the (x, y) coords of the polygon outline. 
    Return a Polygon ogr.Geometry object. 
    
    """
    if isinstance(coords, numpy.ndarray):
        coords = coords.tolist()
    geomDict = {'type':'Polygon', 'coordinates':[coords]}
    geom = ogr.CreateGeometryFromJson(repr(geomDict))
    return geom


def geomFromInteriorPoints(coords):
    """
    The given list of pairs (or 2-d numpy array) is the (x, y) coords of a set of internal
    points inside a polygon. Returns a MultiPoint Geometry. 
    
    """
    if isinstance(coords, numpy.ndarray):
        coords = coords.tolist()
    geomDict = {'type':'MultiPoint', 'coordinates':coords}
    geomPoints = ogr.CreateGeometryFromJson(repr(geomDict))
    return geomPoints


def polygonFromInteriorPoints(geom, preferredEpsg):
    """
    Given a MultiPoint geometry object in lat/long, create a polygon of the 
    convex hull of these points. 
    
    First project the lat/long points into the preferred EPSG, so that when we find 
    the convex hull, we are not crossing any discontinuities such as the international date 
    line. 
    
    Return a single polygon geometry in lat/long. 
    
    """
    (projTr, llTr) = makeTransformations(4326, preferredEpsg)

    geomProj = copyGeom(geom)
    geomProj.Transform(projTr)
    geomOutline = geomProj.ConvexHull()
    geomOutline.Transform(llTr)
    return geomOutline


def crossesDateline(geom, preferredEpsg):
    """
    Given a Polygon Geometry object, in lat/long, detect whether it crosses the dateline. 
    Do this in the projection of the preferred EPSG, so we remove (reduce?) the ambiguity
    about inside/outside. 
    
    """
    (xMin, xMax, yMin, yMax) = geom.GetEnvelope()
    (projTr, llTr) = makeTransformations(4326, preferredEpsg)

    geomProj = copyGeom(geom)
    geomProj.Transform(projTr)
    dateLineGeom = ogr.Geometry(wkt='LINESTRING(180 {}, 180 {})'.format(yMin, yMax))
    try:
        dateLineGeom.Transform(projTr)
        crosses = geomProj.Intersects(dateLineGeom)
    except Exception:
        # If we can't transform into the preferred EPSG, then it seems likely that
        # the geom is nowhere near the date line. 
        crosses = False
    return crosses


def splitAtDateline(geom, preferredEpsg):
    """
    Given a Polygon Geometry object in lat/long, determine whether it crosses the date line, 
    and if so, split it into a multipolygon with a part on either side. 
    
    Use the given preferred EPSG to perform calculations. 
    
    Return a new Geometry in lat/long. 
    
    """
    crosses = crossesDateline(geom, preferredEpsg)
    if crosses:
        (projTr, llTr) = makeTransformations(4326, preferredEpsg)
        coords = getCoords(geom)
        (x, y) = (coords[:, 0], coords[:, 1])
        (yMin, yMax) = (y.min(), y.max())
        xMinPositive = None
        xMaxNegative = None
        xGe0 = (x >= 0)
        xLt0 = (x < 0)
        if xGe0.any() > 0 and xLt0.any() > 0:
            xMaxNegative = x[xLt0].max()
            xMinPositive = x[xGe0].min()
        
            # Create rectangles for the east and west hemispheres, constrained by the 
            # extent of this polygon. Note that this assumes that we do not
            # cross both the date line, and also the prime (zero) meridian. This may not
            # always be true, notably when we are close to the pole. 
            eastHemiRectCoords = [[xMinPositive, yMax], [xMinPositive, yMin], [180, yMin], 
                [180, yMax], [xMinPositive, yMax]]
            eastHemiRectJson = repr({'type':'Polygon', 'coordinates':[eastHemiRectCoords]})
            westHemiRectCoords = [[-180, yMax], [-180, yMin], [xMaxNegative, yMin], 
                [xMaxNegative, yMax], [-180, yMax]]
            westHemiRectJson = repr({'type':'Polygon', 'coordinates':[westHemiRectCoords]})
            eastHemiRect = ogr.CreateGeometryFromJson(eastHemiRectJson)
            westHemiRect = ogr.CreateGeometryFromJson(westHemiRectJson)

            geomProj = copyGeom(geom)
            geomProj.Transform(projTr)
            eastHemiRect.Transform(projTr)
            westHemiRect.Transform(projTr)

            eastHemiPart = geomProj.Intersection(eastHemiRect)
            westHemiPart = geomProj.Intersection(westHemiRect)
            eastHemiPart.Transform(llTr)
            westHemiPart.Transform(llTr)
            
            # Put these together as a single multipolygon
            eastPartCoords = getCoords(eastHemiPart)
            westPartCoords = getCoords(westHemiPart)
            # Discard any vertices which are still no the wrong side of the 180 line. I
            # do not understand what is going on here, but I have invested far more of 
            # my valuable time than I should, and this kludge will be a reasonable approximation. 
            eastPartCoords = eastPartCoords[eastPartCoords[:, 0] > 0, :]
            westPartCoords = westPartCoords[westPartCoords[:, 0] < 0, :]
            
            # Convert to lists
            eastPartCoords = eastPartCoords.tolist()
            westPartCoords = westPartCoords.tolist()
            # Discard anything left with only 2 points
            if len(eastPartCoords) < 3:
                eastPartCoords = []
            if len(westPartCoords) < 3:
                westPartCoords = []
            # Close polygons. What a kludge.....
            if len(eastPartCoords) > 2:
                if eastPartCoords[-1][0] != eastPartCoords[0][0] or eastPartCoords[-1][1] != eastPartCoords[0][1]:
                    eastPartCoords.append(eastPartCoords[0])
            if len(westPartCoords) > 2:
                if westPartCoords[-1][0] != westPartCoords[0][0] or westPartCoords[-1][1] != westPartCoords[0][1]:
                    westPartCoords.append(westPartCoords[0])
            
            # Make a multi-polygon from the two parts
            coordsMulti = [[eastPartCoords], [westPartCoords]]
            jsonStr = repr({'type':'MultiPolygon', 'coordinates':coordsMulti})
            newGeom = ogr.CreateGeometryFromJson(jsonStr)
        else:
            # It didn't really cross the date line, but seems to due to rounding
            # error in crossesDateline(). 
            newGeom = copyGeom(geom)
    else:
        newGeom = copyGeom(geom)
    return newGeom
