import torch
import torch.nn as nn

def patchify(imgs, in_chans, patch_size):
    """
    imgs: (N, c, H, W)
    x: (N, L, patch_size**2 *c)
    """
    c = in_chans
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


class LISTAEmbed(torch.nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim, depth):
        super(LISTAEmbed, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.W_e = nn.ModuleList([nn.Linear(patch_size**2 * in_chans, embed_dim, bias=False) for i in range(depth)])
        self.S = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for i in range(depth)])
        self.theta = nn.Parameter(torch.ones(depth, requires_grad=False))
        self.relu = torch.nn.ReLU()

    def _shrink(self, x, theta):
        return torch.sign(x) * self.relu(torch.abs(x) - theta)

    def forward(self, patches):
        B, num_patches, hw = patches.shape
        Z = torch.zeros((B, num_patches, self.embed_dim))
        for i in range(self.depth):
            C = self.W_e[i](patches) + self.S[i](Z)
            Z = self._shrink(C, self.theta[i])
        return Z


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        """
        imgs: (N, c, H, W)
        x: (N, L, patch_size**2 *c)
        """
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

if __name__ == '__main__':
    N = 10
    in_chans = 3
    img_size = 224
    patch_size = 16
    embed_dim = 1024
    depth = 4
    LISTA_embed = LISTAEmbed(patch_size, in_chans, embed_dim, depth)
    patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
    X = torch.randn(10, 3, img_size, img_size)
    print("Input batch shape:", X.shape)
    XL = LISTA_embed(patchify(X, in_chans, patch_size))
    XP = patch_embed(X)
    print("LISTA embedding batch shape:", XL.shape)
    print("Patch embedding batch shape:", XP.shape)
