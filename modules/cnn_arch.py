"""Shared CNN architecture definition. Imported by both
training/train_cnn.py and modules/cnn_detector.py so training and
inference can never silently drift out of sync with each other (a
common, hard-to-debug source of "trained model doesn't work at
inference time" bugs when the architecture is duplicated by hand).

torch is only imported inside build_model(), so importing this module
doesn't require torch to be installed unless you actually call it.
"""


def build_model(num_classes: int = 2):
    import torch.nn as nn

    def block(cin, cout, stride=2):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class CompactForensicCNN(nn.Module):
        """~1.9M params — deliberately compact so it trains in a
        reasonable time on a single GPU (or even CPU, slowly) without
        needing a pretrained backbone. Trained entirely from random
        initialization; see training/train_cnn.py."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                block(3, 32, 2), block(32, 64, 2), block(64, 128, 2),
                block(128, 128, 1), block(128, 256, 2), block(256, 256, 1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.pool(self.features(x)))

    return CompactForensicCNN()
