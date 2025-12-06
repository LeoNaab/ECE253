import cv2
import numpy as np

img = cv2.imread(r"C:\Users\bscha\Desktop\fa25\ece 253\opaque_obstruction_final6finalv2\01_photos_scaled\banana_scaled.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 30, 120)

# Slightly thicken to make a solid mesh
kernel = np.ones((3,3), np.uint8)
mesh = cv2.dilate(edges, kernel, 1)
inverse_mesh = inverted = cv2.bitwise_not(mesh)


cv2.imwrite("mesh_edges.png", inverse_mesh)