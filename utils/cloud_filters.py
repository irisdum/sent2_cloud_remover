# File of the different functions required to filter the images with clouds on the roi using GEE api
#Python script adapted from http://bit.ly/Sen2CloudFree
import ee
import math
from constant.gee_constant import cloudThresh, erodePixels, dilationPixels, ndviThresh, irSumThresh, roi_ccp_max
ee.Initialize()
cloudHeights = ee.List([i for i in range(200,10250,250)])

def rescale(img, exp, thresholds):
    return img.expression(exp, {"img":img}).subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])


def dilatedErossion(score):
    score = score.reproject('EPSG:4326', None, 20).focal_min(radius=erodePixels, kernelType='circle', iterations=3)\
        .focal_max(radius=dilationPixels, kernelType= 'circle', iterations= 3).reproject('EPSG:4326', None, 20)
    return (score)


def computeS2CloudScore(img):
    #print("in the function")
    toa = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']).divide(10000)
    toa = toa.addBands(img.select(['QA60']))
    score=ee.Image.constant(1)
    #print("score set")
    #print("uuu {}".format(score.serialize()))
    #print("uuu {}".format(score.serialize()))
    score=score.min(rescale(toa, 'img.B2',[0.1, 0.5]))
    #print("uuu {}".format(score.serialize()))
    score = score.min(rescale(toa, 'img.B1', [0.1, 0.3]))
    score = score.min(rescale(toa, 'img.B1 + img.B10', [0.15, 0.2]))
    score = score.min(rescale(toa, 'img.B4 + img.B3 + img.B2', [0.2, 0.8]))
    #print("uuu {}".format(score.serialize()))
    ndmi = img.normalizedDifference(['B8', 'B11'])
    #print("uuu {}".format(score.serialize()))
    score = score.min(rescale(ndmi, 'img', [-0.1, 0.1]))
    ndsi = img.normalizedDifference(['B3', 'B11'])
    score = score.min(rescale(ndsi, 'img', [0.8, 0.6]))
    #print("uuu {}".format(score.serialize()))
    score = score.max(ee.Image.constant(0.001))
    #print("uuu {}".format(score.serialize()))
    dilated = dilatedErossion(score).min(ee.Image(1.0))
    score = score.reduceNeighborhood(reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(5))
    #print("uuu {}".format(score.serialize()))
    return img.addBands(score.rename('cloudScore'))

def calcCloudStats(img):
    imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
        ee.Geometry(img.get('system:footprint')).coordinates()
    )
    roi = ee.Geometry(img.get('ROI'))
    intersection = roi.intersection(imgPoly, ee.ErrorMargin(0.5))
    cloudMask = img.select(['cloudScore']).gt(cloudThresh).clip(roi).rename('cloudMask')
    cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())
    stats = cloudAreaImg.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels= 1e12)
    cloudPercent = ee.Number(stats.get('cloudMask')).divide(imgPoly.area(0.001)).multiply(100)
    coveragePercent = ee.Number(intersection.area()).divide(roi.area(0.001)).multiply(100)
    cloudPercentROI = ee.Number(stats.get('cloudMask')).divide(roi.area(0.001)).multiply(100)
    img = img.set('CLOUDY_PERCENTAGE', cloudPercent)
    img = img.set('ROI_COVERAGE_PERCENT', coveragePercent)
    img = img.set('CLOUDY_PERCENTAGE_ROI', cloudPercentROI)
    return img

def projectShadows(image):
    meanAzimuth = image.get('MEAN_SOLAR_AZIMUTH_ANGLE')
    meanZenith = image.get('MEAN_SOLAR_ZENITH_ANGLE')
    cloudMask = image.select(['cloudScore']).gt(cloudThresh)
    darkPixelsImg = image.select(['B8', 'B11', 'B12']).divide(10000).reduce(ee.Reducer.sum())
    ndvi = image.normalizedDifference(['B8', 'B4'])
    waterMask = ndvi.lt(ndviThresh)
    darkPixels = darkPixelsImg.lt(irSumThresh)
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())
    azR = ee.Number(meanAzimuth).add(180).multiply(ee.Number(math.pi)).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(ee.Number(math.pi)).divide(180.0)

    def find_shadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)
        shadowCastedDistance = zenR.tan().multiply(cloudHeight)
        x = azR.sin().multiply(shadowCastedDistance).multiply(-1)
        y = azR.cos().multiply(shadowCastedDistance).multiply(-1)
        return image.select(['cloudScore']).displace(ee.Image.constant(x).addBands(ee.Image.constant(y)))

    shadows = cloudHeights.map(find_shadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()
    shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))
    shadowScore = shadowMask.reduceNeighborhood(reducer= ee.Reducer.max(),kernel=ee.Kernel.square(1))
    image.addBands(shadowScore.rename(['shadowScore']))
    return image

def computeQualityScore(img):
    score = img.select(['cloudScore']).max(img.select(['shadowScore']))
    score = score.reproject('EPSG:4326', None, 20).reduceNeighborhood(
            reducer=ee.Reducer.mean(),kernel=ee.Kernel.square(5))
    score = score.multiply(-1)
    return img.addBands(score.rename('cloudShadowScore'))


def filter_clouds(sent2_coll,roi):
    """Given a collection of Image from Sentinel 2 MSI product, apply numbers of filters that enables us to return the collection
    sorted from first (less clouds) and remove the images with clouds in the roi.
    :param roi : a ee.Geometry
    :param sent2_coll : a ee.ImageCollection
    :returns : a ee.ImageCollection"""
    print(roi.area(0.001).getInfo())

    s2_coll_filter=sent2_coll.map(lambda img: img.clip(roi))

    s2_coll_filter=s2_coll_filter.map(lambda img: img.set('ROI',roi))

    s2_coll_filter=s2_coll_filter.map(computeS2CloudScore)

    s2_coll_filter=s2_coll_filter.map(calcCloudStats)

    #s2_coll_filter=s2_coll_filter.map(projectShadows)

    #s2_coll_filter=s2_coll_filter.map(computeQualityScore)
   
    s2_coll_filter=s2_coll_filter.sort('CLOUDY_PERCENTAGE_ROI').filter(ee.Filter.lt('CLOUDY_PERCENTAGE_ROI',roi_ccp_max))
    #print("collection filtered {}".format(s2_coll_filter.toList(100).length().getInfo()))
    return s2_coll_filter


