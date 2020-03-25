# Files with all the functions refering to the file hierarchy, how to deal the storing and searchin
import os
import click

def create_hierarchy(output_path):
    """Create an approriate directory directory to download the data"""
    # 1 Check if there is already a directory in output path
    if os.path.isdir(output_path):
        if click.confirm('The directory already exists, it will remove it do you want to continue?', default=True):
            print('Ok remove')
        else:
            return False
    else:
        os.makedirs(output_path) # it is the directory which contains all the images for one area



