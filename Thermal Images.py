import cv2
import numpy as np
import matplotlib.pyplot as plt

# Things to fix:
# White is not fully removed; check rgb and GaussianBlur to fix

# Function to find the cross-hairs that represent avg, max, and min temperatures
def detect_plus_signs(img, size_range=(5, 30)):
    """
    Detect plus-sign shaped markers by color and shape.
    Returns a mask of detected plus signs.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red detection
    lower_red = np.array([0, 200, 200])  # Capture bright, saturated red
    upper_red = np.array([5, 255, 255])  # Narrow hue range to exclude orange tones
    # Might need to go to 4 for upper_red if it doesn't work for certain images

    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # Cyan/green crosshairs
    cyan_mask = cv2.inRange(hsv, np.array([80, 100, 150], dtype=np.uint8),
                            np.array([100, 255, 255], dtype=np.uint8))
    green_mask = cv2.inRange(hsv, np.array([60, 100, 150], dtype=np.uint8),
                             np.array([80, 255, 255], dtype=np.uint8))

    # Process each color separately to apply different rules
    final_mask = np.zeros_like(red_mask)

    # --- CYAN/GREEN PROCESSING ---
    cyan_green = cv2.bitwise_or(cyan_mask, green_mask)
    contours, _ = cv2.findContours(cyan_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Uses the area range: size_range[0] < area < 500
        if size_range[0] < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            # Checks for compact aspect ratio: 0.5 < aspect_ratio < 2.0
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if 0.5 < aspect_ratio < 2.0:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # ------------------
    # --- RED PROCESSING (NOW SAME AS CYAN/GREEN) ---
    # ------------------

    # NOTE: The debug imshow is removed here to align with the cyan/green logic,
    # but you could re-add it if needed.
    cv2.imshow("Red mask before contour finding", red_mask)

    # Use the original red_mask directly
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in red_contours:
        area = cv2.contourArea(cnt)

        # Apply the SAME AREA CHECK as cyan/green: size_range[0] < area < 500
        if size_range[0] < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)

            # Apply the SAME ASPECT RATIO CHECK as cyan/green: 0.5 < aspect_ratio < 2.0
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if 0.5 < aspect_ratio < 2.0:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # Final Dilate (Use a smaller kernel for this final step) - UNCHANGED
    # NOTE: The original had a (5,5) kernel here, but a (3,3) or (5,5) works for dilation.
    final_mask = cv2.dilate(final_mask, np.ones((5, 5), np.uint8), iterations=1)

    return final_mask

# ---- Parameters you can tune ----
input_path = "CH01.jpeg"  # change to your filename
output_clean = "CH01_cleaned_thermal.png"
output_mask = "CH01_thermal_mask.png"

# Location-based removal parameters (adjust based on your thermal camera's UI layout)
SCALE_START_PERCENT = 0.90  # Temperature scale starts at 80% of image width (more aggressive)
CORNER_MARGIN_X_PERCENT = 0.2  # Corner text regions: 35% from left/right edges (increased)
CORNER_MARGIN_Y_PERCENT = 0.08  # Corner text regions: 20% from top/bottom edges (increased)
CROSSHAIR_SIZE_PERCENT = 0.08  # Crosshair region size: 8% of image dimension

# VERY LENIENT thresholds for white/grey/light colors
# Catches colors like #FFF0EB (RGB: 255, 240, 235) and #FEFBFF (RGB: 254, 251, 255)
BGR_LOWER_WHITE = np.array([220, 220, 220], dtype=np.uint8)  # Very light colors
BGR_UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)

# More lenient HSV white detection - very low saturation, high value
# This catches almost-white colors with slight color tints
HSV_LOWER_WHITE = np.array([0, 0, 220], dtype=np.uint8)      # High value (brightness)
HSV_UPPER_WHITE = np.array([180, 30, 255], dtype=np.uint8)   # Low saturation (almost white)

# Thresholds for pale yellow text
HSV_LOWER_YELLOWISH = np.array([20, 10, 200], dtype=np.uint8)
HSV_UPPER_YELLOWISH = np.array([55, 100, 255], dtype=np.uint8)

# Morphology / area filtering
MORPH_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_ITERS = 2
OPEN_ITERS = 1
MIN_AREA_THRESH = 5   # Very small threshold to catch tiny text/symbols
MAX_AREA_THRESH = 8000  # Don't remove very large areas (main thermal data)
DILATE_ITERS = 1  # Reduced to preserve thermal detail
INPAINT_RADIUS = 3

# ---- Load image ----
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Unable to read '{input_path}'")

print(f"Image shape: {img.shape}")

# ---- Build masks ----
# White text (BGR + HSV) - now more lenient
mask_bgr_white = cv2.inRange(img, BGR_LOWER_WHITE, BGR_UPPER_WHITE)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_hsv_white = cv2.inRange(hsv, HSV_LOWER_WHITE, HSV_UPPER_WHITE)

# Pale yellow text
mask_yellowish = cv2.inRange(hsv, HSV_LOWER_YELLOWISH, HSV_UPPER_YELLOWISH)

# Cross-hairs masks
plus_sign_mask = detect_plus_signs(img, size_range = (8,25))

# Combine all masks
mask = cv2.bitwise_or(mask_bgr_white, mask_hsv_white)
mask = cv2.bitwise_or(mask, mask_yellowish)
mask = cv2.bitwise_or(mask, plus_sign_mask)

# ---- Additional: detect very bright pixels that might be text ----
# Very lenient - catches near-white colors like #FFF0EB and #FEFBFF
# THIS IS TOO AGGRESSIVE - threshold too low (210)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- EDGE DETECTION: Catch text on any background (especially over warm colors) ----
# This helps catch white/gray text that appears darker when over red/orange backgrounds
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(gray_blur, 30, 120)

# Dilate edges to make them more prominent
edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

# Only keep edges that are near already detected text (to avoid thermal features)
near_text = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

# Keep edges that are near existing text detection
edges_near_text = cv2.bitwise_and(edges_dilated, near_text)

# --- Restrict to top/bottom regions only ---
h, w = gray.shape[:2]
corner_margin_y_percent = 0.075  # adjust as needed
top_limit = int(h * corner_margin_y_percent*3)
bottom_limit = int(h * (1 - corner_margin_y_percent))
print(top_limit, bottom_limit)

# Create a mask that is 1 only in top/bottom areas
y_mask = np.zeros((h, w), dtype=np.uint8)
y_mask[:top_limit, :] = 255      # top region
y_mask[bottom_limit:, :] = 255   # bottom region

# Apply the y-axis mask
edges_near_text = cv2.bitwise_and(edges_near_text, y_mask)

# Merge with original mask
mask = cv2.bitwise_or(mask, edges_near_text)

# ---- ALTERNATIVE: Detect lighter pixels relative to local background ----
# This catches text that's lighter than surroundings, even if not absolutely bright
kernel_size = 15
local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
difference = gray.astype(np.float32) - local_mean

# Pixels significantly brighter than local area (text on darker thermal regions)
_, bright_relative = cv2.threshold(difference.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)

# ---- Location-based UI element removal ----
h, w = img.shape[:2]
location_mask = np.zeros_like(gray)

# Remove temperature scale bar (right side)
scale_x_start = int(w * SCALE_START_PERCENT)
location_mask[:, scale_x_start:] = 255

# Remove corner number regions
corner_margin_x = int(w * CORNER_MARGIN_X_PERCENT)
corner_margin_y = int(h * CORNER_MARGIN_Y_PERCENT)

# Restrict white/pale-yellow/bright masks to the corners
corner_mask = np.zeros_like(gray)

# Top-left corner
corner_mask[0:int(corner_margin_y*2), 0:corner_margin_x] = 1
# Top-right corner
corner_mask[0:corner_margin_y, w - corner_margin_x:scale_x_start] = 1
# Bottom-right corner
corner_mask[h - corner_margin_y:h, w - int(corner_margin_x//1.25):scale_x_start] = 1
# Bottom-left corner
corner_mask[h - corner_margin_y:h, 0:int(corner_margin_x//2)] = 1

# Only apply in corner regions where text is expected
bright_relative_masked = cv2.bitwise_and(bright_relative, corner_mask)
mask = cv2.bitwise_or(mask, bright_relative_masked)

# Optional: Remove center regions where crosshairs/target info appears
center_x, center_y = w // 2, h // 2
crosshair_size = int(min(w, h) * CROSSHAIR_SIZE_PERCENT)

# Small regions around likely crosshair positions
location_mask[center_y - crosshair_size:center_y + crosshair_size,
center_x - crosshair_size:center_x + crosshair_size] = 255

print(f"Removing regions:")
print(f"  - Temperature scale: x >= {scale_x_start} (right {100 - SCALE_START_PERCENT * 100:.0f}%)")
print(f"  - Corners: {corner_margin_x}x{corner_margin_y} pixels from edges")
print(f"  - Center crosshair: {crosshair_size * 2}x{crosshair_size * 2} pixels")


# ---- Clean mask ----
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=CLOSE_ITERS)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=OPEN_ITERS)

# ---- Keep only reasonably sized components ----
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
clean_mask = np.zeros_like(mask)

areas = []
kept_components = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
    areas.append(area)

    # Keep components that are likely text/UI elements
    if MIN_AREA_THRESH <= area <= MAX_AREA_THRESH:
        # Additional check: aspect ratio for text-like shapes
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio < 15:  # More lenient - was 10
            clean_mask[labels == i] = 255
            kept_components.append((i, area, aspect_ratio))

areas_sorted = sorted(areas, reverse=True)
print(f"Total components found: {len(areas)}")
print(f"Components kept: {len(kept_components)}")
print("Connected component areas (top 10):", areas_sorted[:10])
print("Kept components (label, area, aspect_ratio):", kept_components[:5])

# ---- Light dilation to catch text edges ----
clean_mask = cv2.dilate(clean_mask, np.ones((3, 3), np.uint8), iterations=2)

# NOW add the location mask AFTER component filtering
clean_mask = cv2.bitwise_or(clean_mask, location_mask)

# ---- Inpaint using Navier-Stokes for better edge preservation ----
cleaned = cv2.inpaint(img, clean_mask, INPAINT_RADIUS, cv2.INPAINT_NS)

# ---- Save results ----
cv2.imwrite(output_clean, cleaned)
cv2.imwrite(output_mask, clean_mask)

print(f"Done. Mask saved to {output_mask}, cleaned image saved to {output_clean}")

# ---- Visualization ----
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Top row
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original Thermal Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(mask, cmap='gray')
axs[0, 1].set_title("Combined Raw Mask")
axs[0, 1].axis('off')

axs[0, 2].imshow(clean_mask, cmap='gray')
axs[0, 2].set_title("Final Cleaned Mask")
axs[0, 2].axis('off')

# Bottom row - individual color masks and scale detection
axs[1, 0].imshow(plus_sign_mask, cmap='gray')
axs[1, 0].set_title("Cross-Hair Detection")
axs[1, 0].axis('off')

axs[1, 1].imshow(location_mask, cmap='gray')
axs[1, 1].set_title("Location-based UI Removal")
axs[1, 1].axis('off')

axs[1, 2].imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
axs[1, 2].set_title("Final Cleaned Result")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ---- Optional: Show before/after side by side ----
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original")
ax1.axis('off')

ax2.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
ax2.set_title("Text Removed")
ax2.axis('off')

plt.tight_layout()

# ---- Parameters you can tune ----
input_path = "CH01.jpeg"  # change to your filename
output_clean = "CH01_cleaned_thermal.png"
output_mask = "CH01_thermal_mask.png"

# Location-based removal parameters (adjust based on your thermal camera's UI layout)
SCALE_START_PERCENT = 0.80  # Temperature scale starts at 80% of image width (more aggressive)
CORNER_MARGIN_X_PERCENT = 0.3  # Corner text regions: 30% from left/right edges (increased)
CORNER_MARGIN_Y_PERCENT = 0.20  # Corner text regions: 20% from top/bottom edges (increased)
CROSSHAIR_SIZE_PERCENT = 0.08  # Crosshair region size: 8% of image dimension

# Thresholds for white text
BGR_LOWER_WHITE = np.array([200, 200, 200], dtype=np.uint8)
BGR_UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)

HSV_LOWER_WHITE = np.array([0, 0, 180], dtype=np.uint8)
HSV_UPPER_WHITE = np.array([180, 60, 255], dtype=np.uint8)

# Thresholds for pale yellow text
HSV_LOWER_YELLOWISH = np.array([20, 10, 200], dtype=np.uint8)
HSV_UPPER_YELLOWISH = np.array([55, 100, 255], dtype=np.uint8)

# Thresholds for cyan/green crosshairs and UI elements
HSV_LOWER_CYAN = np.array([80, 100, 150], dtype=np.uint8)
HSV_UPPER_CYAN = np.array([100, 255, 255], dtype=np.uint8)

HSV_LOWER_GREEN = np.array([60, 100, 150], dtype=np.uint8)
HSV_UPPER_GREEN = np.array([80, 255, 255], dtype=np.uint8)

# Morphology / area filtering
MORPH_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_ITERS = 2
OPEN_ITERS = 1
MIN_AREA_THRESH = 20  # Smaller threshold for crosshairs
MAX_AREA_THRESH = 5000  # Don't remove very large areas (main thermal data)
DILATE_ITERS = 1  # Reduced to preserve thermal detail
INPAINT_RADIUS = 3

# ---- Load image ----
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Unable to read '{input_path}'")

print(f"Image shape: {img.shape}")

# ---- Build masks ----
# White text (BGR + HSV)
mask_bgr_white = cv2.inRange(img, BGR_LOWER_WHITE, BGR_UPPER_WHITE)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_hsv_white = cv2.inRange(hsv, HSV_LOWER_WHITE, HSV_UPPER_WHITE)

# Pale yellow text
mask_yellowish = cv2.inRange(hsv, HSV_LOWER_YELLOWISH, HSV_UPPER_YELLOWISH)

# Cyan and green UI elements (crosshairs, etc.)
mask_cyan = cv2.inRange(hsv, HSV_LOWER_CYAN, HSV_UPPER_CYAN)
mask_green = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)

# Combine all masks
mask = cv2.bitwise_or(mask_bgr_white, mask_hsv_white)
mask = cv2.bitwise_or(mask, mask_yellowish)
mask = cv2.bitwise_or(mask, mask_cyan)
mask = cv2.bitwise_or(mask, mask_green)

# ---- Additional: detect very bright pixels that might be text ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_or(mask, bright_mask)

# ---- Location-based UI element removal ----
h, w = img.shape[:2]
location_mask = np.zeros_like(gray)

# Remove temperature scale bar (right side)
scale_x_start = int(w * SCALE_START_PERCENT)
location_mask[:, scale_x_start:] = 255

# Remove corner number regions
corner_margin_x = int(w * CORNER_MARGIN_X_PERCENT)
corner_margin_y = int(h * CORNER_MARGIN_Y_PERCENT)

# Top-left corner (min temp: 49.7, timestamp, etc.)
location_mask[0:corner_margin_y, 0:corner_margin_x] = 255

# Top-right corner (max temp: 71.8) - extend all the way to scale
location_mask[0:corner_margin_y, w - corner_margin_x:scale_x_start] = 255

# Bottom-right corner (additional readings: 49.7) - extend all the way to scale
location_mask[h - corner_margin_y:h, w - corner_margin_x:scale_x_start] = 255

# Optional: Remove center regions where crosshairs/target info appears
center_x, center_y = w // 2, h // 2
crosshair_size = int(min(w, h) * CROSSHAIR_SIZE_PERCENT)

# Small regions around likely crosshair positions
location_mask[center_y - crosshair_size:center_y + crosshair_size,
center_x - crosshair_size:center_x + crosshair_size] = 255

print(f"Removing regions:")
print(f"  - Temperature scale: x >= {scale_x_start} (right {100 - SCALE_START_PERCENT * 100:.0f}%)")
print(f"  - Corners: {corner_margin_x}x{corner_margin_y} pixels from edges")
print(f"  - Center crosshair: {crosshair_size * 2}x{crosshair_size * 2} pixels")


# ---- Clean mask ----
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=CLOSE_ITERS)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=OPEN_ITERS)

# ---- Keep only reasonably sized components ----
# ---- Keep only reasonably sized components ----
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
clean_mask = np.zeros_like(mask)
# ... component filtering code ...

# ---- Light dilation to catch text edges ----
clean_mask = cv2.dilate(clean_mask, np.ones((2, 2), np.uint8), iterations=DILATE_ITERS)

# NOW add the location mask AFTER component filtering
clean_mask = cv2.bitwise_or(clean_mask, location_mask)

# ---- Inpaint using Navier-Stokes for better edge preservation ----
cleaned = cv2.inpaint(img, clean_mask, INPAINT_RADIUS, cv2.INPAINT_NS)
areas = []
kept_components = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
    areas.append(area)

    # Keep components that are likely text/UI elements
    if MIN_AREA_THRESH <= area <= MAX_AREA_THRESH:
        # Additional check: aspect ratio for text-like shapes
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio < 10:  # Avoid very thin lines that might be thermal features
            clean_mask[labels == i] = 255
            kept_components.append((i, area, aspect_ratio))

areas_sorted = sorted(areas, reverse=True)
print(f"Total components found: {len(areas)}")
print(f"Components kept: {len(kept_components)}")
print("Connected component areas (top 10):", areas_sorted[:10])
print("Kept components (label, area, aspect_ratio):", kept_components[:5])

# ---- Light dilation to catch text edges ----
clean_mask = cv2.dilate(clean_mask, np.ones((2, 2), np.uint8), iterations=DILATE_ITERS)

# ---- Inpaint using Navier-Stokes for better edge preservation ----
cleaned = cv2.inpaint(img, clean_mask, INPAINT_RADIUS, cv2.INPAINT_NS)

# ---- Save results ----
cv2.imwrite(output_clean, cleaned)
cv2.imwrite(output_mask, clean_mask)

print(f"Done. Mask saved to {output_mask}, cleaned image saved to {output_clean}")

# ---- Visualization ----
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Top row
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original Thermal Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(mask, cmap='gray')
axs[0, 1].set_title("Combined Raw Mask")
axs[0, 1].axis('off')

axs[0, 2].imshow(clean_mask, cmap='gray')
axs[0, 2].set_title("Final Cleaned Mask")
axs[0, 2].axis('off')

# Bottom row - individual color masks and scale detection
mask_combined_colors = cv2.bitwise_or(mask_cyan, mask_green)
axs[1, 0].imshow(mask_combined_colors, cmap='gray')
axs[1, 0].set_title("Cyan/Green UI Elements")
axs[1, 0].axis('off')

axs[1, 1].imshow(location_mask, cmap='gray')
axs[1, 1].set_title("Location-based UI Removal")
axs[1, 1].axis('off')

axs[1, 2].imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
axs[1, 2].set_title("Final Cleaned Result")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ---- Optional: Show before/after side by side ----
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original")
ax1.axis('off')

ax2.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
ax2.set_title("Text Removed")
ax2.axis('off')

plt.tight_layout()
plt.show()