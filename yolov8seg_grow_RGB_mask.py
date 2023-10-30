from ultralytics import YOLO
import cv2
import numpy as np
import os
# Load a pretrained YOLOv8n model
model = YOLO('/content/drive/MyDrive/grow/models/yolov8n-seg_200e_864p/weights/best.pt')
img=cv2.imread('/content/drive/MyDrive/grow/dataset_1/images/test/2023-10-27_05-54-40.jpeg')
print(img.shape)
# Run inference on 'bus.jpg' with arguments
results=model.predict(img, conf=0.8)

for result in results:
  mask = result.masks.data.cpu().numpy()
  masks = mask.astype(bool)
  mask_shape = masks.shape
  print(mask_shape)

  ori_img = result.orig_img

  # Resize the masks to match the shape of the original image
  resized_masks = [cv2.resize(m.astype(np.uint8), (ori_img.shape[1], ori_img.shape[0])) for m in masks]

  # Define the directory where you want to save the images with contours
  output_dir = "/content/runs/segment/images/"

  # Ensure the output directory exists
  os.makedirs(output_dir, exist_ok=True)

  for i, m in enumerate(resized_masks):
      # Find contours in the binary mask
      contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      # Create a copy of the original image
      img_with_contours = ori_img.copy()
      for j, contour in enumerate(contours):
          # Get the size (area) of the contour
          contour_size = int(cv2.contourArea(contour))
          x, y, w, h = cv2.boundingRect(contour)
          # Create a mask for the contour area
          contour_mask = np.zeros_like(m)
          cv2.drawContours(contour_mask, [contour], -1, 1,thickness=cv2.FILLED)

          # Create a separate image for the contour with only the contour area in RGB
          contour_img = np.zeros_like(ori_img, dtype=np.uint8)
          contour_img[np.where(contour_mask)] = ori_img[np.where(contour_mask)]

          # Set the background of the contour_img to white
          #contour_img[contour_img == 0] = 255
          contour_roi = contour_img[y:y+h, x:x+w]
          #contour_img[:h, :w] = contour_roi
          # Save the contour as an individual image with the size in the filename
          contour_filename = os.path.join(output_dir, f"contour_{i}_{j}_size_{contour_size}.png")
          cv2.imwrite(contour_filename, contour_roi)

          # Print progress for each contour
          print(f"Contour {j} of image {i} with size {contour_size} saved: {contour_filename}")

      # Save the image with all contours to the output directory
      image_filename = os.path.join(output_dir, f"image_with_contours_{i}.png")
      cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # You can adjust the color and thickness


      cv2.imwrite(image_filename, img_with_contours)

      # Print progress for the image with all contours
      print(f"Image {i} with contours saved: {image_filename}")

  print("Images with contours and individual contours saved to", output_dir)
