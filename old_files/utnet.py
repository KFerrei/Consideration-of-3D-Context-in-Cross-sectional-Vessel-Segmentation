import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        b, c, h, w, z = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, ff_dim, dropout_rate):
        super(UNetTransformer, self).__init__()

        self.bottleneck = TransformerBlock(in_channels, num_heads, ff_dim, dropout_rate)

    def forward(self, x):
        x = self.bottleneck(x)
        return x
    
# Exemple de dimensions d'entrée
batch_size = 32   # Nombre d'images dans un batch
in_channels = 256  # Nombre de canaux d'entrée (par exemple, image en niveaux de gris)
out_channels = 1 # Nombre de canaux de sortie
height = 4     # Hauteur de l'image
width = 4      # Largeur de l'image

# Créer une entrée aléatoire
input_tensor = torch.randn(batch_size, in_channels, height, width, 1)

# Instancier le modèle
model = UNetTransformer(in_channels=in_channels, out_channels=out_channels, num_heads=4, ff_dim=512, dropout_rate=0.1)

# Faire passer l'entrée à travers le modèle
output_tensor = model(input_tensor)

# Afficher les dimensions de la sortie
print("Dimensions de l'entrée :", input_tensor.shape)
print("Dimensions de la sortie :", output_tensor.shape)