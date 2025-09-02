#------------------------------------Conversion from ECG image to Time Series Data--------------------------#
# Name: Yehoshua Benjamin Perez Condori
# Student ID: 2231349

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import tensorflow as tf (not used)

# Read the image
image_path = "C:/Users/ybenj/Desktop/Projects/Python/ML/ML/ecgreading.png"
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function to load and preprocess the image
def IMG_Load_and_Preprocess(image_path):
    # Read the image
    image = cv2.imread(image_path)
    processed_image = np.zeros_like(image)  # Create an empty image with the same shape as the original

    # Loop through each pixel to remove the background
    for i in range(image.shape[0]):  # Iterate over the rows
        for j in range(image.shape[1]):  # Iterate over the columns
            # If the pixel value is less than or equal to 230 in all channels, keep it; otherwise, set to white
            if np.all(image[i, j] <= 230):
                processed_image[i, j] = image[i, j]
            else:
                processed_image[i, j] = [255, 255, 255]

    # Convert the processed image to grayscale
    img_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    # Display the grayscale image
    cv2.imshow('Image', img_gray)

    # Create a DataFrame from the grayscale image
    df = pd.DataFrame(img_gray)
    # Plot the image using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(df, cmap='gray')
    plt.colorbar()
    plt.title('Time series')
    plt.axis()  # Disable x and y axes
    plt.show()

    # Replace white (255) with black (0) in the DataFrame
    df.replace(255, 0, inplace=True)
    # Save the DataFrame to a CSV file
    csv_path = 'C:/Users/ybenj/Desktop/Projects/Python/ML/image_data.csv'
    df.to_csv(csv_path)

    # Convert the image to grayscale using OpenCV
    img_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
   
    return img_gray

# Function to apply Gaussian blur
def Gaussian_Blur(img):
    # Apply Gaussian blurring with a 3x3 kernel
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    return blurred_img

# Function to perform Canny edge detection
def Canny_Edge_Detection(blurred_img):
    # Apply Canny edge detection with lower threshold 75 and upper threshold 200
    edges = cv2.Canny(blurred_img, 75, 200)
    return edges

# Function to find contours in the edge-detected image
def Find_Contours(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to draw contours on the image
def Draw_Contours(image, contours):
    # Draw contours on a copy of the image
    contour_img = cv2.drawContours(image.copy(), contours, -1, (255, 0, 0), 2)
    return contour_img

# Function to extract x and y coordinates from contours
def XY_Coordinates_Extraction(contour):
    # Extract x and y coordinates from the contour points
    x_coordinates = [point[0][0] for point in contour]
    y_coordinates = [point[0][1] for point in contour]
    # Sort the coordinates by x-values
    sorted_coordinates = sorted(zip(x_coordinates, y_coordinates))
    sorted_x, sorted_y = zip(*sorted_coordinates)
    return sorted_x, sorted_y

# Function to convert coordinates to time series data
def TimeSeries_Conversion(x_coordinates, y_coordinates):
    # Create time series data as a list of (x, y) tuples
    TimeSeries_Data = list(zip(x_coordinates, y_coordinates))
    return TimeSeries_Data

# Function to plot images
def plot_images(original_image, grayscale_image, blurred_image, edge_image, contour_img):
    titles = ['Original Image', 'Grayscale Image', 'Blurred Image', 'Canny Edge (LT:75 HT:200) Sigma: 0', 'Contour Image']
    images = [original_image, grayscale_image, blurred_image, edge_image, contour_img]

    # Plot each image with its corresponding title
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Remove x and y ticks

    # Show the plot
    plt.show()

# Main function to execute the processing pipeline
def main(image_path):
    # Load and preprocess the image
    preprocessed_image = IMG_Load_and_Preprocess(image_path)
    
    # Apply Gaussian blur
    blurred_img = Gaussian_Blur(preprocessed_image)
    
    # Convert blurred image to uint8 type for thresholding
    blurred_img_uint8 = blurred_img.astype(np.uint8)
    
    # Apply thresholding to convert the image to binary
    ret, thresh = cv2.threshold(blurred_img_uint8, 127, 255, 0)
    
    # Find edges using Canny edge detection
    edges = Canny_Edge_Detection(thresh)
    
    # Find contours in the edge-detected image
    contours = Find_Contours(edges)
    
    # Draw all contours on the original image
    contour_img = Draw_Contours(img_gray, contours)
    
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Apply closing operation to fill gaps in contours
    closed_contour_img = cv2.morphologyEx(contour_img, cv2.MORPH_CLOSE, kernel)
    
    # Plot the images
    plot_images(img_gray, preprocessed_image, blurred_img, edges, closed_contour_img)

# Execute the main function if this script is run directly
if __name__ == "__main__":
    main(image_path)
