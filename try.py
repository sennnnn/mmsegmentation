import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.sample = nn.Unfold(
            kernel_size=(2, 2), dilation=1, padding=0, stride=2)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = x.permute(0, 2, 1).reshape(B, C, H, W).permute(0, 1, 3, 2)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = self.sample(x)

        x = x.view(B, 2, 4)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


N, H, W, C = (1, 4, 4, 2)
x = torch.randint(0, 100, (N, H, W, C))
op = nn.Unfold(kernel_size=(2, 2), dilation=1, padding=0, stride=2)

nchw_x = x.permute(0, 3, 2, 1)
out1 = op(nchw_x.float())
print(out1)
# out1 = out1.permute(0, 2, 1).contiguous()
# out1 = out1.view(N, C * H // 2 * W // 2, 4)
# out1 = out1.permute(0, 2, 1).contiguous()
out1 = out1.view(N, -1, 4 * C).long()

x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
out2 = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
out2 = out2.view(N, -1, 4 * C)  # B H/2*W/2 4*C

out_string = ''
for _ in range(N):
    for i in range(H):
        for j in range(W):
            string = ''
            for k in range(C):
                string += f'{x[_, i, j, k]}-'
            out_string += f'{string} '
        out_string += '\n'
print(out_string)
out_string = ''
for _ in range(N):
    for i in range(out1.shape[1]):
        for j in range(out1.shape[2]):
            out_string += f'{out1[_][i][j]} '
        out_string += '\n'
print(out_string)
out_string = ''
for _ in range(N):
    for i in range(out2.shape[1]):
        for j in range(out2.shape[2]):
            out_string += f'{out2[_][i][j]} '
        out_string += '\n'
print(out_string)
exit(0)
print('out1:', out1)
print('out2:', out2)

exit(0)
