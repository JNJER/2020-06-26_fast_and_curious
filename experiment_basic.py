
#import model's script and set the output file
from DCNN_benchmark.models import *
filename = f'results/{datetag}_results_1_{HOST}.json'

try:
    df = pd.read_json(filename)
except:
    df = pd.DataFrame([], columns=['model', 'perf', 'fps', 'time', 'label', 'i_label', 'i_image', 'filename', 'device']) 
    i_trial = 0
    
    # image preprocessing
    for i_image, (data, label) in enumerate(image_dataset):
        for name in models.keys():
            model = models[name]
            model.eval()
            tic = time.time()
            out = model(data.unsqueeze(0).to(device)).squeeze(0)
            percentage = torch.nn.functional.softmax(out[i_labels], dim=0) * 100
            _, indices = torch.sort(percentage, descending=True)           
            dt = time.time() - tic
            i_label_top = reverse_labels[image_dataset.classes[label]]
            perf_ = percentage[reverse_i_labels[i_label_top]].item()            
            df.loc[i_trial] = {'model':name, 'perf':perf_, 'time':dt, 'fps': 1/dt,
                               'label':labels[i_label_top], 'i_label':i_label_top, 
                               'i_image':i_image, 'filename':image_dataset.imgs[i_image][0], 'device':str(device)}
            print(f'The {name} model get {labels[i_label_top]} at {perf_:.2f} % confidence in {dt:.3f} seconds')
            i_trial += 1
    df.to_json(filename)
