from PIL import Image
import numpy as np

# Load the viridis colormap image
img = Image.open("public/cm_viridis.png")
img_array = np.array(img)

# Create a gradient alpha channel from left to right
width, height = img_array.shape[1], img_array.shape[0]
alpha_gradient = np.linspace(0, 255, width)
alpha = np.tile(alpha_gradient, (height, 1))

# Add alpha channel to the image
rgba = np.concatenate([img_array, alpha[:, :, np.newaxis]], axis=2)

# Convert to uint8 before creating image
rgba = rgba.astype(np.uint8)

# Save the modified image
output_img = Image.fromarray(rgba)
output_img.save("public/cm_transparent_viridis.png", "PNG")
