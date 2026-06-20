import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewScaler(nn.Module):
    """Learnable per-view affine calibration: x * scale + shift."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale + self.shift


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size,
                               padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        # Match channels if needed
        self.project = None
        if in_channels != filters:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, filters, 1, bias=False),
                nn.BatchNorm2d(filters),
            )

    def forward(self, x):
        shortcut = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.project is not None:
            shortcut = self.project(shortcut)

        out = out + shortcut
        out = F.relu(out)
        return out


class SharedEncoder(nn.Module):
    """Input: (N, 2, 24, 24) -> output: (N, 128)."""
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2, bias=False),  # 'same'
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.pool1 = nn.MaxPool2d(2)  # 24 -> 12

        self.block2 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
        )
        self.pool2 = nn.MaxPool2d(2)  # 12 -> 6

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        # GlobalAveragePooling2D
        x = x.mean(dim=(2, 3))  # (N, 128)
        return x


class IvysaurusModel(nn.Module):
    """
    PyTorch translation of IvysaurusIChooseYou.

    NOTE: outputs are raw LOGITS (no softmax) so that nn.CrossEntropyLoss
    can be used directly. Apply softmax at inference time if you need
    probabilities.
    """
    def __init__(self, dimensions, nclasses, nTrackVars, nShowerVars):
        super().__init__()

        self.encoder = SharedEncoder()  # shared across all views & start/end

        # One scaler per view
        self.scalerU = ViewScaler()
        self.scalerV = ViewScaler()
        self.scalerW = ViewScaler()

        # Each branch: concat(start_feat, end_feat) -> 128 + 128 = 256
        # Three branches (U/V/W) -> 256 * 3 = 768
        branch_feat = 128 * 2
        combined_feat = branch_feat * 3 + nTrackVars + nShowerVars

        self.head = nn.Sequential(
            nn.Linear(combined_feat, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.out = nn.Linear(256, nclasses)

    def _branch(self, scaler, start, start_mask, end, end_mask):
        # start/end shapes: (N, 1, H, W); masks same.
        start = torch.cat([start, start_mask], dim=1)  # (N, 2, H, W)
        end   = torch.cat([end, end_mask], dim=1)

        start = scaler(start)
        end   = scaler(end)

        start_feat = self.encoder(start)  # (N, 128)
        end_feat   = self.encoder(end)    # (N, 128)

        return torch.cat([start_feat, end_feat], dim=1)  # (N, 256)

    def forward(self,
                startU, startU_mask, endU, endU_mask,
                startV, startV_mask, endV, endV_mask,
                startW, startW_mask, endW, endW_mask,
                trackVars, showerVars):

        branchU = self._branch(self.scalerU, startU, startU_mask, endU, endU_mask)
        branchV = self._branch(self.scalerV, startV, startV_mask, endV, endV_mask)
        branchW = self._branch(self.scalerW, startW, startW_mask, endW, endW_mask)

        combined = torch.cat([branchU, branchV, branchW,
                              trackVars, showerVars], dim=1)

        combined = self.head(combined)
        logits = self.out(combined)
        return logits
