# Files with all the functions refering to the file hierarchy, how to deal the storing and searchin
import os
import click
from shutil import rmtree
from gee_constant import DIR_T, DIR_SENT, TEMPORARY_DIR, SENT_FORMAT, LISTE_BANDE_S2
import glob

def list_directory(path_downloads):
    """:returns a list of path to the directory which corresponds to all the unzip folders"""
    list_dir=os.listdir(path_downloads)
    return [path_downloads+ path +'/' for path in list_dir]

def list_image(path_image_zip_dir,sent):

    return glob.glob("{}**/*.{}".format(path_image_zip_dir,SENT_FORMAT[sent-1]),recursive=True)


def create_hierarchy(output_path):
    """Create an approriate directory directory to download the data"""
    # 1 Check if there is already a directory in output path
    if os.path.isdir(output_path):
        if click.confirm('The directory already exists, it will remove it do you want to continue?', default=True):
            print('Ok remove')
            #TODO remove the directory tree.rm I think
        else:
            return False
    assert output_path[-1] == "/", "wrong output path directory should end with / but is {}".format(output_path)
    os.makedirs(output_path) # it is the directory which contains all the images for one area
    os.makedirs(output_path + TEMPORARY_DIR)
    mk_sentineldir(output_path + DIR_T[0])
    mk_sentineldir(output_path + DIR_T[1])

def mk_sentineldir(path_time_dir):
    os.makedirs(path_time_dir)
    os.makedirs(path_time_dir + DIR_SENT[0])
    os.makedirs(path_time_dir + DIR_SENT[1])

def combine_image(lpath,output_path_dir):
    """Given a list of path to different image bands : combine them to create one final image
    :returns : list of str the output paths"""
    print("The bands are going to be combined in the following order from top to bottom {}".format(lpath))
    # Combine all the images and store them in the temporary directory
    return "final_combine_path"

def sent2_convert(list_image_path, output_path_dir,lband=None):
    """:param list_image_path : list of path to jpg2 images
    :param lband : list of the bands to be selectec to create the final sentinel image"""
    assert type(list_image_path)==type([]),"Wrong input format should be list but is {}".format(list_image_path)
    if lband is None :
        lband=LISTE_BANDE_S2 #TODO gérer les bandes de sentinel 2

    path_saving_dir= output_path_dir + TEMPORARY_DIR

    #save the image in temporary directory
    return []

def store_image(list_path_image,sent,t,output_path_dir):
    """:param path_image : list path of the images that should be stored with a same fp
    :param sent : 1 or 2 depending of the satellite used,
    :param t : corresponds to the first date or second date of acquisition"""
    pass
    if sent==1:
        assert "vv" in list_path_image[0], "Wrong order of the path in the list path to the VV should always be first " \
                                           "\n{}".format(list_path_image)
        final_path=combine_image(list_path_image)
    else:
        list_converted_image_path=sent2_convert(list_path_image,output_path_dir) #convert all the images into geotiff
        assert len(list_converted_image_path)>0, "Wrong conversion of {} \n paths : {} \n the output path list is " \
                                         "empty".format(sent,list_path_image)
        final_path=combine_image(list_converted_image_path)

    save_image(final_path,output_path_dir,sent,t)
    #Delete the image from the temporary directory

def save_image(final_path, output_path_dir, sent, t):
    """:param final_path : path to the image
    :param: output_path_dir : path to the main directory
    :param sent : int 1 or 2
    :param t : int 1 or 2"""
    #create a copy of the image in the output_dir_path right folders

    #hierarchy path within the outputpath_dir :
    hierarchy_path= DIR_T[sent - 1] + DIR_SENT[sent - 1]
    print("hierarchy path {}".format(hierarchy_path))
    image_name=final_path.split("/")[-1]
    stroring_path=output_path_dir+hierarchy_path+image_name
    os.system("cp {} {}".format(final_path,stroring_path))



def storing_process(output_path_dir,unzipped_dwnld_path_dir,t):
    # First get the list of all the directory downloads
    list_dir=list_directory(unzipped_dwnld_path_dir)
    for dir_image in list_dir :
        #extract sentinel
        sent=path_2_sent(dir_image)
        list_image_path=list_image(dir_image,sent)

    assert len(list_image_path)>0,"No images format {} found in {}".format(SENT_FORMAT[sent - 1], unzipped_path_dir)
    #make the hierarchy
    create_hierarchy(output_path_dir)
    #store the image the images will be converted and band combined to be set in an appropriate directory
    store_image(list_image_path,sent,t,output_path_dir)
    #Delete the temporary directory

def path_2_sent(path_unzipped_dir):
    if "S1" in path_unzipped_dir:
        return 1
    elif "S2" in path_unzipped_dir:
        return 2
    else:
        assert True, "Wrong path to unzipped dir {}".format(path_unzipped_dir)