import math

import torch
import torch.nn as nn

from .pdt import PDT

TARGET_SIZE = 128
CHANNEL = 8
PDT_INPUT_NORM_EPS = 1e-6


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, 1, padding=padding, bias=False)

    def forward(self, x):
        return torch.relu(self.conv(x)) + x


class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return torch.relu(self.conv(x)) + x


def _image_encoder(in_channels, channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        ResBlock2D(channels),
        nn.AvgPool2d(kernel_size=2, stride=2),
        ResBlock2D(channels),
        nn.AvgPool2d(kernel_size=2, stride=2),
        ResBlock2D(channels),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


def _aux_branch(in_channels, channels, kernel_size):
    return nn.Sequential(
        nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding="same", bias=False),
        nn.ReLU(),
        ResBlock(channels, kernel_size=kernel_size, padding="same"),
        nn.AdaptiveMaxPool1d(1),
    )


class TrajVQEncoder(nn.Module):
    def __init__(self, lon_size, lat_size, max_seq_length, image_channels, aux_kernel_sizes,
                 pdt_m=16, pdt_k=256, pdt_steps=1, pdt_heads=1, pdt_input_norm=False):
        super().__init__()
        self.lon_size, self.lat_size = lon_size, lat_size
        self.image_channels = image_channels
        self.pdt_input_norm = pdt_input_norm

        self.MLP = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16))

        self.channel_1d = CHANNEL * 4
        self.conv_new = nn.Sequential(
            nn.Conv1d(16, self.channel_1d, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        for i in range(int(math.log2(max_seq_length)) - 1):
            self.conv_new.add_module("res{}".format(i + 1), ResBlock(self.channel_1d))
            self.conv_new.add_module("pool{}".format(i + 1), nn.MaxPool1d(2))

        channel_2d = CHANNEL // 2
        self.conv_xy = _image_encoder(1, channel_2d)

        lon_length, lat_length = lon_size, lat_size
        for _ in range(4):
            lon_length, lat_length = int(lon_length / 2), int(lat_length / 2)
        total_size = self.channel_1d + lon_length * lat_length * channel_2d
        self.fc1 = nn.Linear(total_size, total_size)
        self.fc2 = nn.Linear(total_size, TARGET_SIZE)
        self.PDT_model = PDT(TARGET_SIZE, pdt_m, pdt_k, pdt_steps, pdt_heads)
        self.norm = nn.LayerNorm(TARGET_SIZE)
        self.register_buffer("pdt_input_mean", torch.zeros(1, TARGET_SIZE))
        self.register_buffer("pdt_input_std", torch.ones(1, TARGET_SIZE))
        self.register_buffer("pdt_input_norm_ready", torch.tensor(False, dtype=torch.bool))

        self.conv_xy = _image_encoder(image_channels, channel_2d)
        self.c_aux = self.channel_1d // 2
        k1, k2 = aux_kernel_sizes
        self.branch_k1 = _aux_branch(16, self.c_aux, k1)
        self.branch_k2 = _aux_branch(16, self.c_aux, k2)
        self.ms_proj = nn.Linear(self.c_aux * 2, self.channel_1d)
        self.ms_alpha_logit = nn.Parameter(torch.tensor(-1.7346))

    def load_backbone_checkpoint(self, checkpoint_path):
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        model_state = self.state_dict()
        matched = {k: v for k, v in checkpoint_state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        self.load_state_dict(model_state)
        return len(matched), len(checkpoint_state) - len(matched)

    def set_pdt_input_normalization(self, mean, std):
        with torch.no_grad():
            self.pdt_input_mean.copy_(mean.view(1, -1))
            self.pdt_input_std.copy_(torch.clamp(std.view(1, -1), min=PDT_INPUT_NORM_EPS))
            self.pdt_input_norm_ready.fill_(True)

    def normalize_pdt_input(self, e):
        if not self.pdt_input_norm or not bool(self.pdt_input_norm_ready.item()):
            return e
        return (e - self.pdt_input_mean.to(e.device)) / self.pdt_input_std.to(e.device)

    def quantize(self, e):
        z = self.PDT_model.encode(self.normalize_pdt_input(e))
        codes = self.PDT_model.get_codes(z)
        z_hat = self.PDT_model.reconstruction(codes)
        e_hat = self.PDT_model.decode(z_hat)
        return {"z": z, "codes": codes, "z_hat": z_hat, "e_hat": e_hat}

    def _encode_sequence(self, x, y):
        stem = self.MLP(torch.cat((x.permute(0, 2, 1), y.permute(0, 2, 1)), dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        n = len(stem)
        x_main = self.conv_new(stem).view(n, self.channel_1d)
        x_aux = torch.cat((self.branch_k1(stem).view(n, self.c_aux), self.branch_k2(stem).view(n, self.c_aux)), dim=1)
        return x_main + torch.sigmoid(self.ms_alpha_logit) * self.ms_proj(x_aux)

    def encode(self, x, y, image):
        n = len(image)
        x_seq = self._encode_sequence(x, y)
        x_img = self.conv_xy(image.contiguous().view(n, self.image_channels, self.lon_size, self.lat_size)).view(n, -1)
        feature = torch.cat((x_seq, x_img), dim=1)
        return self.norm(self.fc2(torch.relu(self.fc1(feature))))
