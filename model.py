"""
Model definitions for the Digits_Classifier live demo.

Both classes are copied from digits_classifier.ipynb without changing layer
order, attribute names, or sizes. That matters: load_state_dict() matches
weights by attribute path (e.g. "convol_layer.0.weight", "model.1.weight"),
so any rename here would break loading the .pth files trained in Colab.
"""
import torch
import torch.nn as nn

MNIST_PIXELS = 28 * 28  # train.data.shape[1] * train.data.shape[2] in the notebook


class MLP(nn.Module):
    # 28*28 -> 256 -> 128 -> 10
    def __init__(self, inputs=MNIST_PIXELS, hiddens=[256, 128], output_classes=10) -> None:
        super().__init__()
        self.layers = []
        self.layers.append(nn.Flatten())

        # add hidden layers
        current = inputs
        for hidden_dim in hiddens:
            self.layers.append(nn.Linear(current, hidden_dim))
            self.layers.append(nn.ReLU())
            current = hidden_dim

        # add output layers
        self.layers.append(nn.Linear(current, output_classes))
        # define steps
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class CNN_model(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), number_classes=10):
        super().__init__()

        self.convol_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (16, 14, 14)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Shape: (32, 7, 7)
        )

        # input_dim calculation
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            input_dim = self.convol_layer(dummy_input).view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, number_classes),
        )

    def forward(self, x):
        conv_output = self.convol_layer(x)
        return self.fc_layers(conv_output)


def load_model(model_cls, weights_path, device="cpu"):
    """Rebuild the architecture, load the Colab state_dict, switch to eval mode.

    eval() is required, not optional: BatchNorm1d raises on a batch of one
    image in training mode, and Dropout(0.5) would make predictions random.
    """
    model = model_cls()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
