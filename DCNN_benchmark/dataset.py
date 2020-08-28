
from DCNN_benchmark.init import *

list_dir = os.listdir(path)

if os.path.isdir(path):
    print("The folder " , folder, " already exists, it includes: ", list_dir)   # check if the folder exist and create one 
    
else :
    print(f"No existing path match for this folder, creating a folder at {path}")
    os.makedirs(path)

print(list_dir)
if len(list_dir) < N_labels : # if there aren't anough labels download some more
    print('This folder do not have anough classes, downloading some more') # using the downloader
    cmd =f"python3 ImageNet-datasets-downloader/downloader.py -data_root {root} -data_folder {folder} -images_per_class {N_images_per_class} -use_class_list True  -class_list {id_dl} -multiprocessing_workers 0"
    print('Command to run : ', cmd)
    os.system(cmd) # running it
    list_dir = os.listdir(path)
    
elif len(os.listdir(path)) == N_labels :
    print(f'The folder already contains : {len(list_dir)} classes')
          
else : # if there are to many folders delete some
    print('The folder have to many classes, deleting some')
    while len(os.listdir(path)) > N_labels : 
        for elem in os.listdir(path):
            contenu = os.listdir(f'{path}/{elem}')
            for x in contenu:
                os.remove(f'{path}/{elem}/{x}') # delete exces folders
        try:
            os.rmdir(f'{path}/{elem}')
        except:
            os.remove(f'{path}/{elem}')
    list_dir = os.listdir(path)
    print("Now the folder " , folder, f" contains :", os.listdir(path)) 
