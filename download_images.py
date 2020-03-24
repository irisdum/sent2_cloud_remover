# Functions to download images
import io
from io import StringIO ## for Python 3

from auscophub.saraclient import searchSara, makeUrlOpener, makeQueryUrl, readJsonUrl
import pandas as pd
import requests
import zipfile


def download_images(url_images):
    import requests
    if url_images is None:
        url_images="/thredds/fileServer/fj7/Copernicus/Sentinel-1/C-SAR/GRD/2019/2019-12/00N095E-05S100E/S1A_IW_GRDH_1SDV_20191206T231322_20191206T231338_030234_0374D3_511C.png"
    print('Beginning file download with requests')
    r = requests.get(url_images)

    with open('.jpg', 'wb') as f:
        f.write(r.content)

    # Retrieve HTTP meta-data
    print(r.status_code)
    print(r.headers['content-type'])
    print(r.encoding)


def reformat_dataframe(df):
    """:param df: a pandas Dataframe which contains all the features
    :returns the dataframe with two new columns : one with the name of the image and the other with the download
    link"""
    df["image_id"]= df.apply (lambda row: row["properties"]["title"], axis=1)
    #print(df.head)
    df["download_path"]=df.apply (lambda row: get_zip_path(row), axis=1)
    return df


def get_zip_path(row):
    """given a df row returns the path to download the zip folder of the image of the row"""
    png_path=row["properties"]["quicklook"]
    zip_path=png_path[:-3]+"zip"
    return zip_path

def get_image_download_path(df,image_id):
    """Given the df and the image_id
    :returns the path to the download"""
    #print(df["image_id"])
    #print(image_id)
    zip_path=df["download_path"][df["image_id"]==image_id]
    return zip_path

def transformfilter2query(filter_name,filter_value):
    print(filter_name+"="+filter_value)
    return filter_name+"="+filter_value


def get_download_zip_url(image_name, sent=1, dict_param=None):
    if dict_param is None:
        dict_param = {"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"}
    proxy = None
    urlOpener = makeUrlOpener(proxy)
    lparam=[]
    for key in dict_param:
        lparam+=[transformfilter2query(key,dict_param[key])]
    allfeatures=searchSara(urlOpener,sent,lparam)
    df = pd.DataFrame(allfeatures)
    df = reformat_dataframe(df)
    final_df=get_image_download_path(df,image_name)
    if final_df.size==0:
        print("NO image found with {}".format(lparam))
        return None
    else:
        print(get_image_download_path(df, image_name).iloc[0])
        return get_image_download_path(df, image_name).iloc[0]


def download_url(zip_file_url):

    #print(r)
    r = requests.get(zip_file_url, stream=True)
    print(r)
    print(io.BytesIO(r.content))
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print(z)
    z.extractall()


def main():
    proxy=None
    urlOpener = makeUrlOpener(proxy)
    image_id_test="S1B_IW_GRDH_1SSH_20200322T220406_20200322T220435_020810_027767_839F"
    #dict_param={"startDate":"2020-03-22","completionDate":"2020-03-23","productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"}
    dict_param2={"startDate": "2020-03-22", "completionDate": "2020-03-23"}
    zip_url=get_download_zip_url(image_id_test, 2, dict_param2)
    output_path="/root/code/sent2-cloud-remover/test_data/"+image_id_test+".zip"
    download_url(zip_url)
    #allFeatures=searchSara(urlOpener,1,dict_param)
    #print(allFeatures)
    #df=pd.DataFrame(allFeatures)
    #df = pd.DataFrame(results["features"])
    #df = reformat_dataframe(df)
    #print(get_image_download_path(df, image_id_test).iloc[0])


if __name__ == '__main__':
    main()
