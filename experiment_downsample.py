
#import model's script and set the output file
from DCNN_benchmark.models import *
filename = f'results/{datetag}_results_2_{HOST}.json'

# Output's set up
try:
    df_downsample = pd.read_json(filename)
except:
    df_downsample = pd.DataFrame([], columns=['model', 'perf', 'fps', 'time', 'label', 'i_label', 'i_image', 'image_size', 'filename', 'device']) 
    i_trial = 0

    # image preprocessing
    for image_size in image_sizes:
        image_size = int(image_size)
        transform = transforms.Compose([  # Downsampling function on the input
        transforms.Resize(image_size),      #  Resize the image to image_size x image_size pixels size.
        transforms.CenterCrop(image_size),  # Crop the image to image_size x image_size pixels around the center.
        transforms.ToTensor(),       # Convert the image to PyTorch Tensor data type.
        transforms.Normalize(        # Normalize the image by adjusting its average and
                                     # its standard deviation at the specified values.
        mean=[0.485, 0.456, 0.406],                
        std=[0.229, 0.224, 0.225]                  
        )])
        image_dataset_downsample = ImageFolder(path, transform=transform) # Get the downsample dataset
        print(f'Résolution de {image_size}')
        # Displays the input image of the model 
        for i_image, (data, label) in enumerate(image_dataset_downsample):
            for name in models.keys():
                model = models[name]
                model.eval()
                tic = time.time()
                out = model(data.unsqueeze(0).to(device)).squeeze(0)
                percentage = torch.nn.functional.softmax(out[i_labels], dim=0) * 100
                _, indices = torch.sort(percentage, descending=True)           
                dt = time.time() - tic
                i_label_top = reverse_labels[image_dataset_downsample.classes[label]]
                perf_ = percentage[reverse_i_labels[i_label_top]].item()            
                df_downsample.loc[i_trial] = {'model':name, 'perf':perf_, 'time':dt, 'fps': 1/dt,
                                   'label':labels[i_label_top], 'i_label':i_label_top, 
                                   'i_image':i_image, 'filename':image_dataset.imgs[i_image][0], 'image_size': image_size, 'device':str(device)}
                print(f'The {name} model get {labels[i_label_top]} at {perf_:.2f} % confidence in {dt:.3f} seconds')
                i_trial += 1
        df_downsample.to_json(filename)
