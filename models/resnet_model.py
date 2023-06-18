import sys
sys.path.append("..")  # Adds higher directory to python modules path.

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from .common_layers import Stage

class ResnetModel(nn.Module):
    def __init__(self, mode, weights=None, *args, **kwargs):
        super(ResnetModel, self).__init__(*args, **kwargs)
        
        # Get the pre-trained ResNet50 model
        resnet50 = models.resnet50(pretrained=True).cuda()
        summary(resnet50, (3, 28, 28))
        
        self.stages = []
        layers = []

        # Stage 1: Convolution and max pooling layers
        stage = torch.nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )
        print('===> Stage 1 <===')
        print(stage)
        print()
        layers += stage
        self.stages.append(Stage(64, stage))

        # Stage 2: First residual block
        stage = resnet50.layer1
        print('===> Stage 2 <===')
        print(stage)
        print()
        layers += stage
        self.stages.append(Stage(256, stage))

        # Stage 3: Second residual block 
        stage = resnet50.layer2
        print('===> Stage 3 <===')
        print(stage)
        print()
        layers += stage
        self.stages.append(Stage(512, stage))

        # Stage 4: Third residual block
        stage = resnet50.layer3
        print('===> Stage 4 <===')
        print(stage) 
        print()
        layers += stage
        self.stages.append(Stage(1024, stage))

        # Stage 5: Fourth residual block 
        stage = resnet50.layer4
        print('===> Stage 5 <===')
        print(stage)
        print()
        layers += stage
        self.stages.append(Stage(2048, stage))
        self.stages = nn.ModuleList(self.stages)

        self.features = nn.Sequential(*layers)
        self.avg_pool = resnet50.avgpool
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

        if mode == 'gender_classifier':
            # gender_classifier_head = [
            #     nn.BatchNorm1d(resnet50.fc.in_features),
            #     nn.Dropout(0.2),
            #     nn.Linear(resnet50.fc.in_features, 512),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(512),
            #     nn.Dropout(0.2),
            #     nn.Linear(512, 2),
            #     # nn.Softmax(),
            # ]

            gender_classifier_head = [
                nn.Dropout(0.2),
                nn.Linear(resnet50.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2),
                # nn.Softmax(),
            ]

            # gender_classifier_head = [
            #     nn.Linear(resnet50.fc.in_features, 4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 2),
            # ]

            self.head = nn.Sequential(*gender_classifier_head)
        elif mode == 'age_regressor':
            age_regressor_head = [
                nn.Dropout(0.2),
                nn.Linear(resnet50.fc.in_features, 1),
                nn.ReLU(),
            ]
            self.head = nn.Sequential(*age_regressor_head)  
        elif mode == 'age_classifier':
            
            # age_classifier_head = [
            #     nn.BatchNorm1d(resnet50.fc.in_features),
            #     nn.Dropout(0.2),
            #     nn.Linear(resnet50.fc.in_features, 512),
            #     nn.ReLU(),
            #     nn.BatchNorm1d(512),
            #     nn.Dropout(0.2),
            #     nn.Linear(512, 100),
            #     # nn.Softmax(),
            # ]

            age_classifier_head = [
                nn.Dropout(0.2),
                nn.Linear(resnet50.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 100),
                # nn.Softmax(),
            ]

            # age_classifier_head = [
            #     nn.Linear(resnet50.fc.in_features, 4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU(),
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 100),
            # ]

            self.head = nn.Sequential(*age_classifier_head)  

        print(f'===> {mode} Head <===')
        print(self.head)
        print()

        self.weights = weights
        self.init_weights()
        self.mode = mode

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.head(x)
        if self.mode == 'age_regressor':
            return x
        else:
            return x, self.softmax(x)

    def init_weights(self):
        for layer in self.head.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        print('-' * 30)
        print(f'===> Load {self.weights} Weights <===')

        weight_path = None
        if self.weights == 'gender_classifier':
            weight_path = 'ckpts/gender_classifier ckpt.pth'        
            pretrained_dict = torch.load(weight_path)
        elif self.weights == 'age_regressor':
            weight_path = 'ckpts/age_regressor ckpt.pth'
            pretrained_dict = torch.load(weight_path)
        elif self.weights == 'age_classifier':
            weight_path = 'ckpts/age_classifier ckpt.pth'
            pretrained_dict = torch.load(weight_path)

        key_matching_num = 0
        if weight_path is not None:
            print('weight_path:', weight_path)

            pretrained_dict = pretrained_dict['model_state_dict']

            print('len(pretrained_dict.keys()):', len(pretrained_dict.keys()))
            print('len(self.state_dict().keys()):', len(self.state_dict().keys()))
            for key in list(pretrained_dict.keys()):
                if key in self.state_dict().keys():
                    key_matching_num += 1
            
            self.load_state_dict(pretrained_dict)

        print('key_matching_num:', key_matching_num)
        print('-' * 30)
        
        
        
