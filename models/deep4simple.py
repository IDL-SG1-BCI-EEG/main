import torch
import torch.nn as nn

class Deep4Network(nn.Module):
    def __init__(self, in_channels, n_classes, conv_stride = 3, batch_norm_alpha=0.1, drop_prob=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            # Conv Pool Block 1
            nn.Conv2d(in_channels, 25, (10, 1)),
            nn.Conv2d(25, 25, (1, in_channels), stride=(conv_stride, 1), bias=False),
            nn.BatchNorm2d(25, momentum=batch_norm_alpha, affine=True, eps=1e-5,),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),

            # Conv Pool Block 2
            nn.Dropout(p=drop_prob),
            nn.Conv2d(25, 50, (10, 1), bias=False),
            nn.BatchNorm2d(50, momentum=batch_norm_alpha),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),

            # Conv Pool Block 3
            nn.Dropout(p=drop_prob),
            nn.Conv2d(50, 100, (10, 1), bias=False),
            nn.BatchNorm2d(100, momentum=batch_norm_alpha),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),

            # Conv Pool Block 4
            nn.Dropout(p=drop_prob),
            nn.Conv2d(100, 200, (10, 1), bias=False),
            nn.BatchNorm2d(200, momentum=batch_norm_alpha),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),

            # Classification Layer
            nn.Conv2d(200, n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

