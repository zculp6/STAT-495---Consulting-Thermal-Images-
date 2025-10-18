library(jpeg)
library(EBImage)
library(OpenImageR)

img <- readJPEG("CH01.jpeg")
img
dim(img)


img <- readImage("CH01.jpeg")
display(img)

gray <- channel(img, "gray")
# Example: mask only top-left corner where numbers appear
mask <- matrix(FALSE, nrow=nrow(gray), ncol=ncol(gray))

# Define bounding boxes manually (row/col ranges)
mask[1:40, 1:100] <- TRUE     # top-left text block
mask[200:240, 1:80] <- TRUE   # bottom-left timestamp
mask[1:100, 220:320] <- TRUE  # top-right temp scale

mask_clean <- opening(mask, makeBrush(5, shape='disc'))  # remove small specks
mask_clean <- closing(mask_clean, makeBrush(5, shape='disc')) # fill small gaps

# Apply median filter to masked areas
img_matrix <- imageData(img)
for (ch in 1:3) {
  channel_img <- img_matrix[,,ch]
  channel_img[mask] <- median(channel_img[!mask])  # Replace with median of surroundings
  img_matrix[,,ch] <- channel_img
}

clean_img <- Image(img_matrix, colormode = Color)
display(clean_img)
