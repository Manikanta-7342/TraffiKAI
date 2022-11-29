from PIL import Image,ImageFilter

import torchvision.transforms as transforms
from torchvision import *
from torch import *
import cv2 as cv2
import time
#from modelclass import *


from torchvision import *

from torch import *

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        # _,out = torch.max(out,dim = 1)
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)
        return loss

    # def validation_step(self, batch):
    #     images, targets = batch
    #     out = self(images)
    #
    #     # Generate predictions
    #     loss = F.binary_cross_entropy(torch.sigmoid(out), targets)
    #
    #     score = accuracy(out, targets)
    #     return {'val_loss': loss.detach(), 'val_score': score.detach()}

    # this 2 methods will not change .

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


class Densenet169(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.pretrained_model = models.densenet169(pretrained=True)

        feature_in = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(feature_in, 2)

    def forward(self, x):
        return self.pretrained_model(x)

def emergency_image(east_path,south_path,west_path,north_path,model_path):
    imsize = (512, 512)
    frames_to_skip = 10
    li = []
    im_output="Test/"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    # loading the model
    loaded_densenet169 = Densenet169()
    loaded_densenet169.load_state_dict(torch.load(model_path+'densenet169.pt', map_location=torch.device('cpu')))
    loaded_densenet169.eval()

    def predict_emergency_vehicle(image_path):
        uploaded_file = image_path

        image = Image.open(uploaded_file).convert('RGB')

        image = image.filter(ImageFilter.MedianFilter)

        image = transform(image).view(1, 3, 512, 512)

        pred = loaded_densenet169.forward(image)
        proba, idx = torch.max(torch.sigmoid(pred), dim=1)

        proba = proba.detach().numpy()[0]
        idx = idx.numpy()[0]

        #li.append(float(proba)) if idx == 1 else li.append(1 - float(proba))
        li.append(float(proba)) if idx == 1 else li.append(1-float(proba))

    video_reader_east = cv2.VideoCapture(east_path)
    video_reader_west = cv2.VideoCapture(west_path)
    video_reader_north = cv2.VideoCapture(north_path)
    video_reader_south = cv2.VideoCapture(south_path)
    tick = time.time()
    try:
        while True:
            ret, image = video_reader_east.read()
            image = cv2.resize(image, imsize)
            cv2.imwrite(im_output+'east.png', image)

            ret, image = video_reader_west.read()
            image = cv2.resize(image, imsize)
            cv2.imwrite(im_output+'west.png', image)

            ret, image = video_reader_north.read()
            image = cv2.resize(image, imsize)
            cv2.imwrite(im_output+'north.png', image)

            ret, image = video_reader_south.read()
            image = cv2.resize(image, imsize)
            cv2.imwrite(im_output+'south.png', image)

        for k in range(frames_to_skip):
            video_reader_east.grab()
            video_reader_west.grab()
            video_reader_north.grab()
            video_reader_south.grab()
    except:
        pass



    predict_emergency_vehicle(im_output+"east.png")
    predict_emergency_vehicle(im_output+"south.png")
    predict_emergency_vehicle(im_output+"west.png")
    predict_emergency_vehicle(im_output+"north.png")

    #print('\n\n', 'Time taken: ', time.time() - tick)
    video_reader_east.release()
    video_reader_west.release()
    video_reader_north.release()
    video_reader_south.release()

    return li

