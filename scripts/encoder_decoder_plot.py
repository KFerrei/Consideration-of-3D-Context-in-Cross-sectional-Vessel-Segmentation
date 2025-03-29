import torch
import torch.nn as nn
import torchvision.transforms as transforms
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Définition de l'encodeur et du décodeur avec convolution
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 1, 128, 128)  # Reshape pour correspondre aux dimensions image
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded.view(-1, 128 * 128)

# Charger une image NIfTI et la transformer
def load_nifti(image_path):
    img = nib.load(image_path)
    img_data = img.get_fdata()
    img_slice = img_data[:, :, img_data.shape[2] // 2]  # Prendre une coupe au milieu
    img_slice = np.interp(img_slice, (img_slice.min(), img_slice.max()), (0, 1))  # Normalisation
    img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)  # Ajouter une dimension de canal
    return img_tensor

def save_image(tensor, filename):
    image = tensor.view(128, 128).detach().numpy()
    plt.imsave(filename, image, cmap='gray')

# Charger l'image
torch.manual_seed(42)  # Pour la reproductibilité
image_path = "/home/kefe11/ThesisProject/data/Dataset_25mm_128p_9s/train/images/AF066_Examination1_Slice1_LEFT.nii.gz"
image_tensor = load_nifti(image_path)
print(image_tensor.shape)

# Instancier le modèle
model = Autoencoder()

# Définir l'optimiseur et la fonction de perte
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement sur une seule image
epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()
    encoded, decoded = model(image_tensor)
    loss = criterion(decoded, image_tensor.view(-1, 128 * 128))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")


encoded, decoded = model(image_tensor)

# Sauvegarder l'image encodée et décodée
save_image(image_tensor, "image.png")
save_image(decoded, "decoded.png")
save_image(encoded, "encoded.png")
print("Image encodée et décodée sauvegardée !")