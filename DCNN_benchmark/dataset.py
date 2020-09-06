
from DCNN_benchmark.init import *

# check if the folder exist
if os.path.isdir(path):
    list_dir = os.listdir(path)
    print("The folder " , folder, " already exists, it includes: ", list_dir)
    
# no folder, creating one 
else :
    print(f"No existing path match for this folder, creating a folder at {path}")
    os.makedirs(path)

# if the folder is empty, download the images using the ImageNet-Datasets-Downloader
if len(list_dir) < N_labels : 
    print('This folder do not have anough classes, downloading some more') 
    cmd =f"python3 ImageNet-Datasets-Downloader/downloader.py -data_root {root} -data_folder {folder} -images_per_class {N_images_per_class} -use_class_list True  -class_list {id_dl} -multiprocessing_workers 0"
    print('Command to run : ', cmd)
    os.system(cmd) # running it
    list_dir = os.listdir(path)
    
elif len(os.listdir(path)) == N_labels :
    print(f'The folder already contains : {len(list_dir)} classes')
          
else : # if there are to many folders delete some
    print('The folder have to many classes, deleting some')
    for elem in os.listdir(path):
        contenu = os.listdir(f'{path}/{elem}')
        if len(os.listdir(path)) > N_labels :
            for x in contenu:
                os.remove(f'{path}/{elem}/{x}') # delete exces folders
            try:
                os.rmdir(f'{path}/{elem}')
            except:
                os.remove(f'{path}/{elem}')
    list_dir = os.listdir(path)
    print("Now the folder " , folder, f" contains :", os.listdir(path))
