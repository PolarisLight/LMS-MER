import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=10):
        super().__init__()
        self.backbone_name = backbone.lower()
        
        # --------------------------------------------------------
        # ResNet family
        # --------------------------------------------------------
        if self.backbone_name == "resnet18":
            net = models.resnet18(weights=None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()

        elif self.backbone_name == "resnet50":
            net = models.resnet50(weights=None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()

        # --------------------------------------------------------
        # VGG16 configuration
        # --------------------------------------------------------
        elif self.backbone_name == "vgg16":
            net = models.vgg16(weights=None)
            # Use 1x1 adaptive pooling so flattened features stay compact.
            net.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Remove the original classifier head and expose backbone features.
            net.classifier = nn.Identity()
            
            # Feature dimension after global pooling.
            feat_dim = 512 

        # --------------------------------------------------------
        # DenseNet121
        # --------------------------------------------------------
        elif self.backbone_name == "densenet121":
            net = models.densenet121(weights=None)
            feat_dim = net.classifier.in_features
            net.classifier = nn.Identity()

        # --------------------------------------------------------
        # MobileNet V2
        # --------------------------------------------------------
        elif self.backbone_name == "mobilenet_v2":
            net = models.mobilenet_v2(weights=None)
            # MobileNetV2 classifier is Sequential; the last layer is Linear.
            feat_dim = net.classifier[-1].in_features
            net.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = net
        
        # feat_dim is inferred from the selected backbone.
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feat=False):
        # 1) Extract backbone features.
        feat = self.backbone(x)
        
        # 2) Ensure features are flattened to [N, C].
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
            
        # 3) Classify.
        logits = self.classifier(feat)
        
        if return_feat:
            return logits, feat
        return logits
