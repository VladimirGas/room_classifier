import gradio as gr
import numpy as np
import pandas as pd
import PIL

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import gradio as gr



def load_cond_model(model_path):
    loaded_model = models.resnet152(pretrained=True)
    num_ftrs = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_ftrs, len(COND_NAMES))

    state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    loaded_model.load_state_dict(state_dict)
    return loaded_model



MODEL_PATHS = {'general':'models/general_best_acc_086.pth','kitchen':'models/kitchen_best_acc_073.pth','bathroom':'models/bathroom_best_acc_076.pth'}
TYPE_NAMES = sorted(pd.read_csv('labels/allin_labels.csv').roomType.unique())
COND_NAMES = sorted(pd.read_csv('labels/bathroom_labels.csv').condition.unique())


loaded_models = {key: torch.load(value,map_location=torch.device('cpu')).eval() for key, value in MODEL_PATHS.items()}

transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# PREDICTION 

def predict(img_path):
    img = PIL.Image.open(img_path)
    img = transform(img).unsqueeze(0)
    room_pred = loaded_models['general'](img)
    room_probs = F.softmax(room_pred,dim=1)[0]
    room_result = {TYPE_NAMES[i]:float(room_probs[i]) for i in range(len(TYPE_NAMES))}
    
    if room_result['kitchen']>0.7:
        cond_pred = loaded_models['kitchen'](img)
        cond_prob = F.softmax(cond_pred,dim=1)[0]
        cond_result = {COND_NAMES[i]:float(cond_prob[i]) for i in range(len(COND_NAMES))}
        
    elif room_result['bathroom']>0.7:
        cond_pred = loaded_models['bathroom'](img)
        cond_prob = F.softmax(cond_pred,dim=1)[0]
        cond_result = {COND_NAMES[i]:float(cond_prob[i]) for i in range(len(COND_NAMES))}
        
    else:
        cond_result = None
    
    return room_result, cond_result


# GRADIO UI

title = "Room Classifier"
description = "This app firstly classifies the type of the room (living room, bathroom, kitchen, bedroom, exterior, etc.) and in the cases of kitchens and bathrooms classifies whether the room is in bad, average or good condition."
samples = [[f'sample/sample{i}.jpg'] for i in range(30)]
outputs = [gr.outputs.Label(type="confidences",num_top_classes=4),
           gr.outputs.Label(type="confidences")]

gr.Interface(fn=predict,
             inputs=gr.inputs.Image(type="file"),
             outputs=outputs,
             examples=samples,
             title=title,
             description=description).launch(share=True)














