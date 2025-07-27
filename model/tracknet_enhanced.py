import torch
import torch.nn as nn


class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()

        self.encoder_block1 = self._make_encoder_block(15, 64, 2)
        self.encoder_block2 = self._make_encoder_block(64, 128, 2)
        self.encoder_block3 = self._make_encoder_block(128, 256, 3)
        self.encoder_block4 = self._make_encoder_block(256, 512, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.gru = nn.GRU(input_size=512, hidden_size=512, batch_first=True)

        self.decoder_block1 = self._make_decoder_block(768, 256, 3)
        self.decoder_block2 = self._make_decoder_block(384, 128, 2)
        self.decoder_block3 = self._make_decoder_block(192, 64, 2)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.output_conv = nn.Conv2d(64, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_encoder_block(self, in_channels, out_channels, num_convs):
        layers = []
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels, out_channels, num_convs):
        layers = []
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.encoder_block1(x)
        enc1_pool = self.pool(enc1)

        enc2 = self.encoder_block2(enc1_pool)
        enc2_pool = self.pool(enc2)

        enc3 = self.encoder_block3(enc2_pool)
        enc3_pool = self.pool(enc3)

        bottleneck = self.encoder_block4(enc3_pool)

        B, C, H, W = bottleneck.shape
        temporal_features = bottleneck.view(B, C, H * W).permute(0, 2, 1)
        gru_out, _ = self.gru(temporal_features)
        bottleneck = gru_out.permute(0, 2, 1).view(B, C, H, W)

        dec1 = self.upsample(bottleneck)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.decoder_block1(dec1)

        dec2 = self.upsample(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder_block2(dec2)

        dec3 = self.upsample(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.decoder_block3(dec3)

        output = self.output_conv(dec3)
        output = self.sigmoid(output)

        return output


def generate_heatmap(size, center, sigma=5):
    H, W = size
    x, y = center
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    X = X.float()
    Y = Y.float()
    heatmap = torch.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    return heatmap


if __name__ == "__main__":
    model = TrackNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    batch_size = 2
    test_input = torch.randn(batch_size, 15, 288, 512)
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    gt_heatmaps = torch.zeros(batch_size, 3, 288, 512)
    for b in range(batch_size):
        for f in range(3):
            ball_x = torch.randint(50, 462, (1,)).item()
            ball_y = torch.randint(50, 238, (1,)).item()
            heatmap = generate_heatmap((288, 512), (ball_x, ball_y))
            gt_heatmaps[b, f] = heatmap

    print(f"Ground truth shape: {gt_heatmaps.shape}")
    print(f"GT range: [{gt_heatmaps.min():.3f}, {gt_heatmaps.max():.3f}]")
