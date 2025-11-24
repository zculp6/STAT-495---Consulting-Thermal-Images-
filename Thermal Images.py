import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


# ================================================================
# Detect crosshair coordinates (RED and CYAN/GREEN)
# ================================================================
def detect_plus_sign_coordinates(img):
    """
    Detect plus-sign cross-hairs for red and cyan/green markers.
    Returns:
        red_points -> list of (x, y)
        cyan_points -> list of (x, y)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red detection
    lower_red1 = np.array([0, 215, 80])
    upper_red1 = np.array([6, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([174, 215, 80])
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Cyan + Green
    cyan_mask = cv2.inRange(hsv, np.array([80, 100, 150]), np.array([100, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 100, 120]), np.array([75, 255, 255]))

    # Helper: find centroids of contours
    def extract_points(mask, min_area=8, max_area=500):
        pts = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pts.append((cx, cy))
        return pts

    red_points = extract_points(red_mask)
    cyan_points = extract_points(cyan_mask)
    green_points = extract_points(green_mask)

    return red_points, cyan_points, green_points


# ================================================================
# Existing function (UNCHANGED) – but still used for mask pipeline
# ================================================================
def detect_plus_signs(img, size_range=(5, 30)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red detection
    lower_red1 = np.array([0, 215, 80])
    upper_red1 = np.array([6, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([174, 215, 80])
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Cyan/green detection
    cyan_mask = cv2.inRange(hsv, np.array([80, 157, 20]), np.array([100, 255, 255]))
    lower_green_wide = np.array([40, 50, 50])
    upper_green_wide = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green_wide, upper_green_wide)

    final_mask = np.zeros_like(red_mask)

    # Cyan/Green
    green = cv2.bitwise_or(cyan_mask, green_mask)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if size_range[0] < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if 0.5 < aspect_ratio < 2.0:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    # Red
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if size_range[0] < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if 0.1 < aspect_ratio < 2.0:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return cv2.dilate(final_mask, np.ones((5, 5), np.uint8), iterations=1)


def choose_single_red_point(img, red_points):
    """
    Select exactly one red crosshair point.
    Uses contour area to identify the main crosshair, and then selects the closest point from the input list.
    """
    if len(red_points) == 0:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Use improved red mask (wider range)
    # First range (0 degrees):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # Second range (180 degrees):
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_area = 0

    # 1. FIND THE CENTER OF THE LARGEST RED REGION
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_area = area
                best_center = (cx, cy)

    # 2. CHOOSE A TARGET POINT BASED ON THE BEST CENTER
    target_point = None
    if best_center is not None:
        # Use the largest contour's centroid as the reference point
        ref_cx, ref_cy = best_center
    else:
        # If no contour was found (fallback), use the image center as the reference
        h, w = img.shape[:2]
        ref_cx, ref_cy = w // 2, h // 2

    # Select the point from red_points closest to the reference point
    red_points_sorted = sorted(
        red_points,
        key=lambda p: (p[0] - ref_cx) ** 2 + (p[1] - ref_cy) ** 2
    )
    return red_points_sorted[0]


def process_single_image(input_path, output_dir, visualize=False):
    """
    Process a single thermal image and save outputs.

    Args:
        input_path: Path to input image
        output_dir: Directory to save outputs
        visualize: Whether to show visualization plots
    """
    # ================================================================
    # PARAMETERS
    # ================================================================
    SCALE_START_PERCENT = 0.90
    CORNER_MARGIN_X_PERCENT = 0.2
    CORNER_MARGIN_Y_PERCENT = 0.12

    BGR_LOWER_WHITE = np.array([180, 180, 180], dtype=np.uint8)
    BGR_UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)
    HSV_LOWER_WHITE = np.array([0, 0, 200], dtype=np.uint8)
    HSV_UPPER_WHITE = np.array([180, 40, 255], dtype=np.uint8)
    BGR_LOWER_GRAY = np.array([100, 100, 100], dtype=np.uint8)
    BGR_UPPER_GRAY = np.array([180, 180, 180], dtype=np.uint8)
    HSV_LOWER_YELLOWISH = np.array([20, 10, 200], dtype=np.uint8)
    HSV_UPPER_YELLOWISH = np.array([55, 100, 255], dtype=np.uint8)

    MORPH_KERNEL = np.ones((3, 3), np.uint8)
    CLOSE_ITERS = 2
    OPEN_ITERS = 1
    MIN_AREA_THRESH = 5
    MAX_AREA_THRESH = 8000
    INPAINT_RADIUS = 3

    # ================================================================
    # Load input
    # ================================================================
    img = cv2.imread(input_path)
    if img is None:
        print(f"Unable to read '{input_path}' - skipping")
        return False

    # Get filename without extension
    filename = Path(input_path).stem

    print(f"Processing {filename}... (shape: {img.shape})")

    # ================================================================
    # DETECT CROSSHAIR COORDINATES
    # ================================================================
    h, w = img.shape[:2]

    # 1. Detect raw points on the ORIGINAL image (img)
    # This ensures colors are pure and detectable
    red_points_raw, cyan_points, green_points = detect_plus_sign_coordinates(img)

    # 2. Define the exclusion zone (temperature scale)
    # Use the same parameters used for the location_mask
    scale_x_start = int(w * SCALE_START_PERCENT)

    def is_outside_scale(point, scale_start_x):
        """Returns True if the point is NOT in the temperature scale region."""
        x, y = point
        # The exclusion region is defined by location_mask[:, scale_x_start:]
        return x < scale_start_x

    # 3. Filter points to exclude those on the temperature scale
    red_points_filtered = [p for p in red_points_raw if is_outside_scale(p, scale_x_start)]
    cyan_points_filtered = [p for p in cyan_points if is_outside_scale(p, scale_x_start)]
    green_points_filtered = [p for p in green_points if is_outside_scale(p, scale_x_start)]

    # 4. Choose the single correct red crosshair from the filtered list
    red_single = choose_single_red_point(img, red_points_filtered)

    # 5. Save coordinates to CSV
    coord_rows = []

    # Add SINGLE red point
    if red_single is not None:
        coord_rows.append({"Color": "Red", "X": red_single[0], "Y": red_single[1]})

    # Add ALL cyan points
    for (x, y) in cyan_points_filtered:
        coord_rows.append({"Color": "Cyan", "X": x, "Y": y})

    for (x, y) in green_points_filtered:
        coord_rows.append({"Color": "Green", "X": x, "Y": y})

    df_coords = pd.DataFrame(coord_rows)
    csv_path = os.path.join(output_dir, f"{filename}_coordinates.csv")
    df_coords.to_csv(csv_path, index=False)

    # ================================================================
    # MASK CREATION (original logic)
    # ================================================================
    mask_bgr_white = cv2.inRange(img, BGR_LOWER_WHITE, BGR_UPPER_WHITE)
    mask_bgr_gray = cv2.inRange(img, BGR_LOWER_GRAY, BGR_UPPER_GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv_white = cv2.inRange(hsv, HSV_LOWER_WHITE, HSV_UPPER_WHITE)
    mask_yellowish = cv2.inRange(hsv, HSV_LOWER_YELLOWISH, HSV_UPPER_YELLOWISH)
    plus_sign_mask = detect_plus_signs(img, size_range=(5, 25))

    mask = cv2.bitwise_or(mask_bgr_white, mask_hsv_white)
    mask = cv2.bitwise_or(mask, mask_bgr_gray)
    mask = cv2.bitwise_or(mask, mask_yellowish)
    mask = cv2.bitwise_or(mask, plus_sign_mask)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 30, 120)
    edges_dilated = cv2.dilate(edges, np.ones((4, 4), np.uint8), iterations=1)

    near_text = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    edges_near_text = cv2.bitwise_and(edges_dilated, near_text)

    h, w = gray.shape[:2]
    top_limit = int(h * 0.075 * 3)
    bottom_limit = int(h * (1 - 0.075))
    y_mask = np.zeros((h, w), dtype=np.uint8)
    y_mask[:top_limit, :] = 255
    y_mask[bottom_limit:, :] = 255
    edges_near_text = cv2.bitwise_and(edges_near_text, y_mask)

    mask = cv2.bitwise_or(mask, edges_near_text)

    kernel_size = 15
    local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    difference = gray.astype(np.float32) - local_mean
    _, bright_relative = cv2.threshold(difference.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)

    location_mask = np.zeros_like(gray)
    scale_x_start = int(w * SCALE_START_PERCENT)
    location_mask[:, scale_x_start:] = 255

    corner_mask = np.zeros_like(gray)
    corner_margin_x = int(w * CORNER_MARGIN_X_PERCENT)
    corner_margin_y = int(h * CORNER_MARGIN_Y_PERCENT)
    corner_mask[0:int(corner_margin_y * 2), 0:corner_margin_x] = 1
    corner_mask[0:corner_margin_y, w - corner_margin_x:scale_x_start] = 1
    corner_mask[h - corner_margin_y:h, w - int(corner_margin_x // 1.25):scale_x_start] = 1
    corner_mask[h - corner_margin_y:h, 0:int(corner_margin_x // 2)] = 1

    bright_relative_masked = cv2.bitwise_and(bright_relative, corner_mask)
    mask = cv2.bitwise_or(mask, bright_relative_masked)

    # Define custom masking region: X=198 to w, Y=262 to 675 for minimum temperature at bottom of scale
    X_START = 198
    Y_START = 262
    Y_END = 675

    location_mask_custom_region = np.zeros_like(gray)

    # Ensure bounds are within image limits
    y_slice = slice(Y_START, min(Y_END, h))
    x_slice = slice(X_START, w)

    location_mask_custom_region[y_slice, x_slice] = 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=CLOSE_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=OPEN_ITERS)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w0, h0 = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
        if MIN_AREA_THRESH <= area <= MAX_AREA_THRESH:
            aspect_ratio = max(w0, h0) / max(min(w0, h0), 1)
            if aspect_ratio < 15:
                clean_mask[labels == i] = 255

    clean_mask = cv2.dilate(clean_mask, np.ones((3, 3), np.uint8), iterations=2)
    clean_mask = cv2.bitwise_or(clean_mask, location_mask)

    clean_mask = cv2.bitwise_or(clean_mask, location_mask_custom_region)

    cleaned = cv2.inpaint(img, clean_mask, INPAINT_RADIUS, cv2.INPAINT_NS)

    # Save outputs
    output_clean = os.path.join(output_dir, f"{filename}_cleaned.png")

    cv2.imwrite(output_clean, cleaned)

    print(f"  ✓ Saved: {filename}_cleaned.png, {filename}_coordinates.csv")

    # Visualization (optional)
    if visualize:
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        # Top row
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title(f"Original - {filename}")

        # Mark detected crosshairs
        if not df_coords.empty:
            # Plot RED (one point)
            if red_single is not None:
                axs[0, 0].plot(red_single[0], red_single[1], 'o', color='red',
                               markersize=12, markeredgecolor='white', markeredgewidth=2)
                axs[0, 0].text(red_single[0], red_single[1] - 15,
                               f"({red_single[0]},{red_single[1]})",
                               color='white', fontsize=8, ha='center',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

            # Plot CYAN points
            for (x, y) in cyan_points:
                axs[0, 0].plot(x, y, 'o', color='cyan',
                               markersize=12, markeredgecolor='white', markeredgewidth=2)
                axs[0, 0].text(x, y - 15, f"({x},{y})",
                               color='white', fontsize=8, ha='center',
                               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))

            # Plot GREEN points
            for (x, y) in green_points:
                axs[0, 0].plot(x, y, 'o', color='green',
                               markersize=12, markeredgecolor='white', markeredgewidth=2)
                axs[0, 0].text(x, y - 15, f"({x},{y})",
                               color='white', fontsize=8, ha='center',
                               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))

        axs[0, 0].axis('off')

        axs[0, 1].imshow(mask, cmap='gray')
        axs[0, 1].set_title("Combined Raw Mask")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(clean_mask, cmap='gray')
        axs[0, 2].set_title("Final Cleaned Mask")
        axs[0, 2].axis('off')

        # Bottom row
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
        plt.savefig(os.path.join(output_dir, f"{filename}_analysis.png"), dpi=150)
        plt.close()

    return True


# ================================================================
# BATCH PROCESSING
# ================================================================
def process_folder(input_folder="Cuneo_Hall", output_folder="Cuneo_Hall_cleaned", visualize=False):
    """
    Process all images in the input folder and save to output folder.

    Args:
        input_folder: Path to folder containing thermal images
        output_folder: Path to folder for cleaned outputs
        visualize: Whether to generate visualization plots for each image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Get all image files
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return

    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in '{input_folder}'")
        return

    print(f"\n{'=' * 60}")
    print(f"Found {len(image_files)} images in '{input_folder}'")
    print(f"Output directory: '{output_folder}'")
    print(f"{'=' * 60}\n")

    # Process each image
    success_count = 0
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        if process_single_image(str(img_file), output_folder, visualize):
            success_count += 1
        print()

    print(f"\n{'=' * 60}")
    print(f"Batch processing complete!")
    print(f"Successfully processed: {success_count}/{len(image_files)} images")
    print(f"Results saved to: '{output_folder}'")
    print(f"{'=' * 60}\n")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    # Process all images in Cuneo_Hall folder
    # Set visualize=True to generate analysis plots for each image
    process_folder(
        input_folder="Cuneo_Hall",
        output_folder="Cuneo_Hall_cleaned",
        visualize=True  # Set to True if you want visualization plots
    )