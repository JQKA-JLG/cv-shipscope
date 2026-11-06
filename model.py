import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # B = batch size; e.g. 64
        # Each block applies convolution, ReLU and spatial downsampling.
        self.features = nn.Sequential(
            # Block 1: (B, 3, 80, 80) -> (B, 16, 40, 40)
            # Learn 16 feature maps from 3 RGB input channels.
            # padding=1, stride=1 preserve height and width. Still 80x80
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # Adding nonlinearity by replacing negative values with zero.
            nn.ReLU(),
            # Maximum of 2x2 region, halving height and with. -> 40x40
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (B, 16, 40, 40) -> (B, 32, 20, 20)
            # Learn new features by combining the previous feature maps.
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (B, 32, 20, 20) -> (B, 64, 10, 10)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # Flatten all 64 feature maps per image into a vector of 6400 values.
            nn.Flatten(),

            # 64 maps with 10x10 values = 6400 feature values -> combined into 64 hidden activations.
            nn.Linear(64 * 10 * 10, 64),
            nn.ReLU(),

            # Produces 2 class scores (logits):
            # 0 = no ship, 1 = ship
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Map input batch of shape (B, 3, 80, 80) to logits (B, 2).
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    import torch

    # Check the output shape using 4 randomly generated RGB images.
    model = Net()
    images = torch.rand(4, 3, 80, 80)
    logits = model(images)

    print(logits.shape) # Expected stdout: torch.Size([4, 2])