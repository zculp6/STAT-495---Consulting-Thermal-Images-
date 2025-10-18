import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('CH01_snipped.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV color space for better color detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Target color: rgb(235, 30, 11) - average red of plus sign
# Convert to HSV for reference: H=4, S=95%, V=92%

# Define narrow range around the target red
# HSV values: Hue ~4, high saturation, high value
lower_red = np.array([0, 200, 200])  # Capture bright, saturated red
upper_red = np.array([5, 255, 255])  # Narrow hue range to exclude orange tones

# Create mask
mask = cv2.inRange(hsv, lower_red, upper_red)

# Alternative approach using RGB color distance
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
target_rgb = np.array([235, 30, 11])  # rgb(235, 30, 11)

# Calculate color distance from target
color_diff = np.sqrt(np.sum((rgb_img - target_rgb) ** 2, axis=2))
# Mask pixels within threshold distance (adjust threshold as needed)
rgb_mask = (color_diff < 40).astype(np.uint8) * 255

# Combine both masks (use whichever works better)
mask = cv2.bitwise_or(mask, rgb_mask)

# Optional: Clean up the mask with morphological operations
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply the mask to extract the plus sign
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')
axes[1].axis('off')

axes[2].imshow(result)
axes[2].set_title('Masked Result')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Save the mask if needed
cv2.imwrite('plus_sign_mask.png', mask)
cv2.imwrite('plus_sign_extracted.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))