import cv2
import numpy as np

# Read the image
img = cv2.imread("shape.jpg")
if img is None:
    print("Error: Image not found!")
    exit()

# Resize image
img = cv2.resize(img, (800, 600))

# 1. Image preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# 2. Contour detection
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# Function to calculate angles
def is_right_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)

    cosine = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    )

    return abs(cosine) < 0.25


# Function to classify shapes
def detect_shape(contour):

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour, 0.02 * perimeter, True
    )

    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:

        points = approx.reshape(4, 2)

        angles = [
            is_right_angle(
                points[i - 1],
                points[i],
                points[(i + 1) % 4]
            )
            for i in range(4)
        ]

        sides = [
            np.linalg.norm(points[i] - points[(i + 1) % 4])
            for i in range(4)
        ]

        if all(angles):
            if max(sides) / (min(sides) + 1e-10) < 1.15:
                return "Square"
            else:
                return "Rectangle"

        return "Quadrilateral"

    elif vertices == 5:
        return "Pentagon"

    elif vertices == 6:
        return "Hexagon"

    else:
        area = cv2.contourArea(contour)

        circularity = (
            4 * np.pi * area / (perimeter * perimeter + 1e-10)
        )

        if circularity > 0.80:
            return "Circle"

        return "Polygon"


# 3. Process contours and calculate measurements

for contour in contours:

    area = cv2.contourArea(contour)

    # Filter small contours/noise
    if area < 500:
        continue

    perimeter = cv2.arcLength(contour, True)

    shape = detect_shape(contour)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Draw contour
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

    # Draw bounding box
    cv2.rectangle(
        img, (x, y), (x + w, y + h),
        (255, 0, 0), 2
    )

    # Display shape name
    cv2.putText(
        img, shape,
        (x, y - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2
    )

    # Display area
    cv2.putText(
        img, f"Area: {area:.0f}",
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 0, 0), 2
    )

    print("--------------------------------")
    print("Shape:", shape)
    print("Area:", area)
    print("Perimeter:", round(perimeter, 2))
    print("Bounding Box:", x, y, w, h)


# 4. Display results
cv2.imshow("Shape Detection", img)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Threshold Image", thresh)
cv2.imshow("Edge Detection", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
