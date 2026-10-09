import numpy as np
from PIL import Image

# Encryption key
K = np.array([[3, 2],
              [1, 1]], dtype=int)

# Inverse key modulo 256
K_inv = np.array([[1, 254],
                  [255, 3]], dtype=int)

# Read image as a pixel matrix
image = np.array(
    Image.open("sample_input.png").convert("L"),
    dtype=np.int64
)

h, w = image.shape

# Pad odd dimensions to complete 2x2 blocks
padded = np.pad(
    image,
    ((0, h % 2), (0, w % 2)),
    mode="constant"
)

encrypted = np.zeros_like(padded)
decrypted = np.zeros_like(padded)

# Encryption: C = K B mod 256
for i in range(0, padded.shape[0], 2):
    for j in range(0, padded.shape[1], 2):
        B = padded[i:i+2, j:j+2]
        encrypted[i:i+2, j:j+2] = (K @ B) % 256

# Decryption: B = K_inv C mod 256
for i in range(0, padded.shape[0], 2):
    for j in range(0, padded.shape[1], 2):
        C = encrypted[i:i+2, j:j+2]
        decrypted[i:i+2, j:j+2] = (K_inv @ C) % 256

# Save output images
Image.fromarray(encrypted.astype(np.uint8)).save("encrypted.png")
Image.fromarray(
    decrypted[:h, :w].astype(np.uint8)
).save("decrypted.png")

print("Decryption successful:",
      np.array_equal(image, decrypted[:h, :w]))