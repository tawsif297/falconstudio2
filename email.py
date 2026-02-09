import cv2
import numpy as np

# List of images
images = ["logo1.png", "logo.png"]

for img_name in images:
    # Read image
    img = cv2.imread(img_name)

    # Create mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Rectangle (slightly inside the image)
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)

    # Apply grabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Prepare final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Apply mask
    result = img * mask2[:, :, None]

    # Save result
    out_name = f"output_{img_name}"
    cv2.imwrite(out_name, result)
    print(f"Saved {out_name}")