import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):

    def __init__(self, model, pretrained=True):
        super(CNN, self).__init__()

        assert model in ['alexnet', 'resnet18', 'squeezenet', '']
        self._model = model

        if model == 'alexnet':
            self.model = models.__dict__['alexnet'](pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[0])  ## Get conv part
            self.final_feature_size = 256 * 1 * 7
        elif model == 'resnet18':
            self.model = models.__dict__['resnet18'](pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children()))[:-4]
            self.final_feature_size = 128 * 9 * 24
        elif model == 'squeezenet':  ## TODO (shape mismatch)
            self.model = models.__dict__['squeezenet1_1'](pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children()))[:-1]
            self.final_feature_size = 512 * 4 * 11
        elif model == '':
            self.model = nn.Sequential(
                                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                         )      
            self.final_feature_size = 32 * 18 * 64

    def forward(self, x):
        b, n_channel, n_height, n_width = x.size()
        x = self.model(x)
        #print(x.size()) ## To decide self.final_feature_size
        x = x.view(b, self.final_feature_size)

        return x

class Architecture(nn.Module):

    def __init__(self, model, n_classes, pretrained=True):
        super(Architecture, self).__init__()

        self.n_classes = n_classes

        self.cnn = CNN(model, pretrained=pretrained)
        if model == 'squeezenet':
            self.classifier = nn.Sequential(
                                     nn.Dropout(p=0.5),
                                     nn.Conv2d(self.cnn.final_feature_size, self.n_classes, kernel_size=(1, 1), stride=(1, 1)),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1))
                              )
        else:
            self.classifier = nn.Sequential(
                                     nn.Dropout(0.5),
                                     nn.Linear(self.cnn.final_feature_size, self.n_classes)
                              )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)

        return x
