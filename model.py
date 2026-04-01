# model.py
import torch
import torch.nn as nn


VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)
IMG_SIZE    = 224


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2, use_se=True):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)
        self.pool  = nn.MaxPool2d(2, 2)
        self.se = None
        if use_se:
            r = max(out_ch // 16, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(out_ch, r), nn.ReLU(inplace=True),
                nn.Linear(r, out_ch), nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.block(x)
        if self.se is not None:
            s = self.se(x).view(x.size(0), x.size(1), 1, 1)
            x = x * s
        return self.pool(x)


class MultiLabelVOCNet(nn.Module):
    """
    5 Conv Blocks + SE Attention + GAP
    Input : (B, 3, 224, 224)
    Output: (B, 20) logits — apply sigmoid for probabilities
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.block1 = ConvBlock(  3,  32, depth=2, use_se=False)  # 224→112
        self.block2 = ConvBlock( 32,  64, depth=2, use_se=True)   # 112→56
        self.block3 = ConvBlock( 64, 128, depth=2, use_se=True)   #  56→28
        self.block4 = ConvBlock(128, 256, depth=2, use_se=True)   #  28→14
        self.block5 = ConvBlock(256, 256, depth=2, use_se=True)   #  14→7
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.head(self.gap(x))


def load_model(weights_path: str, device: str = "cpu") -> MultiLabelVOCNet:
    """Load trained model from checkpoint or weights-only file."""
    model = MultiLabelVOCNet(num_classes=NUM_CLASSES)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded weights from {weights_path}")
    return model
