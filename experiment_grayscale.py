
#import model's script and set the output file
from DCNN_benchmark.models import *
filename = f'results/{datetag}_results_3_{HOST}.json'

# Output's set up
try:
    df_gray = pd.read_json(filename)
except:
    df_gray = pd.DataFrame([], columns=['model', 'perf', 'fps', 'time', 'label', 'i_label', 'i_image', 'filename', 'device']) 
    i_trial = 0
    
    # image preprocessing
    transform = transforms.Compose([
    transforms.Grayscale(3),      # convert the image in grayscale
    transforms.Resize(int(image_size)),      # Resize the image.
    transforms.CenterCrop(int(image_size-20)), # Crop the image with a 20 pixels border.
    transforms.ToTensor(),       # Convert the image to PyTorch Tensor data type.
    transforms.Normalize(        # Normalize the image by adjusting its average and
                                 #     its standard deviation at the specified values.
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
    )])
    image_dataset_grayscale = ImageFolder(path, transform=transform) # Get the downsample dataset

    # Displays the input image of the model
    for i_image, (data, label) in enumerate(image_dataset_grayscale):
            for name in models.keys():
                model = models[name]
                model.eval()
                tic = time.time()
                out = model(data.unsqueeze(0).to(device)).squeeze(0)
                percentage = torch.nn.functional.softmax(out[i_labels], dim=0) * 100
                _, indices = torch.sort(percentage, descending=True)           
                dt = time.time() - tic
                i_label_top = reverse_labels[image_dataset_grayscale.classes[label]]
                perf_ = percentage[reverse_i_labels[i_label_top]].item()            
                df_gray.loc[i_trial] = {'model':name, 'perf':perf_, 'time':dt, 'fps': 1/dt,
                                   'label':labels[i_label_top], 'i_label':i_label_top, 
                                   'i_image':i_image, 'filename':image_dataset.imgs[i_image][0], 'device':str(device)}
                print(f'The {name} model get {labels[i_label_top]} at {perf_:.2f} % confidence in {dt:.3f} seconds')
                i_trial += 1
    df_gray.to_json(filename)
