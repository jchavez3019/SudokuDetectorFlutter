from matplotlib import pyplot as plt
import numpy as np
import cv2
from copy import copy
import os
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data.dataset import Subset
from torch import Tensor, Size
from typing import Union, Tuple


def detectSudokuPuzzle(image_path: str, silent: bool = True):
    """
    Extracts a warped and cropped sudoku puzzle from an image.
    @param image_path:
    @param silent:
    @return:
    """
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    colored_image = copy(original_image)
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_width = gray_image.shape[1]
    img_height = gray_image.shape[0]

    if not silent:
        print(f"Shape of image: {img_width}, {img_height}")

    modified_image = cv2.medianBlur(gray_image, 3)
    modified_image = cv2.adaptiveThreshold(modified_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    modified_image = cv2.Canny(modified_image, 250, 200)
    modified_image = cv2.dilate(modified_image, np.ones((3, 3), np.uint8), np.zeros(gray_image.shape), [-1, 1])

    contours, _ = cv2.findContours(modified_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest contour
    contour_areas = np.array([cv2.contourArea(cont) for cont in contours])
    area_indices = np.flip(np.argsort(contour_areas))

    # Just to analyze the area of the largest contour
    largest_area = contour_areas[area_indices[0]]
    total_area = img_width * img_height

    # filter out contours whose area is nearly the entire image which would indicate that there is a border around
    # the image
    l_i = 0
    area_threshold = 0.98
    for i, area_idx in enumerate(area_indices):
        curr_area = contour_areas[area_idx]
        if curr_area / total_area < area_threshold:
            l_i = i
            break
    area_indices = area_indices[l_i:]

    for a_idx in area_indices:
        perimeter = cv2.arcLength(contours[a_idx], True) * 0.02
        approx = cv2.approxPolyDP(contours[a_idx], perimeter, True)

        contourArea = cv2.contourArea(approx)
        if len(approx) == 4 and abs(contourArea) > 2000 and cv2.isContourConvex(approx):
            if not silent:
                print(f"This is a valid contour; Area of contour is {contourArea}; Shape of approx is {approx.shape}")
            for i in range(len(approx)):
                point = np.squeeze(approx[i])
                if not silent:
                    print(f"Point_{i}: ({point[0]},{point[1]})")

            temp_approx = np.squeeze(approx)
            if not silent:
                print(f"temp_approx shape {temp_approx.shape}")
            abs_vals = np.array([temp_approx[i, 0] * temp_approx[i, 1] for i in range(temp_approx.shape[0])])
            tl_idx = np.argmin(abs_vals)
            br_idx = np.argmax(abs_vals)
            top_left = temp_approx[tl_idx, :]
            bottom_right = temp_approx[br_idx, :]
            other_idx = [i for i in range(len(temp_approx)) if (i != tl_idx and i != br_idx)]
            top_right = temp_approx[other_idx[0], :] if temp_approx[other_idx[0], 1] > temp_approx[
                other_idx[1], 1] else temp_approx[other_idx[1], :]
            bottom_left = temp_approx[other_idx[0], :] if temp_approx[other_idx[0], 0] > temp_approx[
                other_idx[1], 0] else temp_approx[other_idx[1], :]

            reordered_approx = np.expand_dims(np.array([top_left, top_right, bottom_right, bottom_left]), axis=1)
            if not silent:
                print(f"Reordered: \n{reordered_approx}")
            approx = reordered_approx
            break

    if img_height < img_width:
        orientation = 'landscape'
        square_size = int(img_height * 0.90)
    else:
        orientation = 'portrait'
        square_size = int(img_width * 0.90)

    # get the perspective transform matrix
    target_coordinates = np.array([[0, 0], [0, square_size], [square_size, square_size], [square_size, 0]],
                                  dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(np.squeeze(approx).astype(np.float32), target_coordinates)

    # Apply the perspective transformation
    result_img = cv2.warpPerspective(colored_image, transformation_matrix, (int(square_size), int(square_size)))

    if not silent:
        print(f"Orientation: {orientation}, Square Size: {square_size}, Image dimensions: {original_image.shape}")
        print(f"Target (x,y) coordinates:\n{target_coordinates}")

    # display results
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1), plt.imshow(colored_image, 'gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_img, 'gray')
    plt.title('Warped Image')
    plt.xticks([]), plt.yticks([])
    plt.show()

    return result_img


def parse_dat(dat_path: str) -> np.ndarray:
    """
    Parses a .dat file and returns the labels of a sudoku puzzle as a 1D np array
    @param dat_path:
    @return:
    """
    numbers = []
    with open(dat_path, 'r') as file:
        for i, line in enumerate(file):
            if i < 2:
                continue
            # Split the string into substrings based on spaces
            number_strings = line.split()

            # Convert each substring to an integer and store them in a list
            numbers.append([int(num_str) for num_str in number_strings])

    return np.array(numbers)

def format_inputs_cells(cells: list[np.ndarray]) -> Tensor:
    cells = np.array(
        [cv2.resize(cell, (50, 50)).astype(np.float32) / 255.0 for cell in
         cells])
    all_images = torch.tensor(cells)
    all_images = all_images.unsqueeze(1)  # add a channel dimension to indicate to the NN the images are black & white

    return all_images

def get_sudoku_dataset(split: float = 0.8, verbose: bool = False) -> Tuple[Subset, Subset, Size]:
    """
    Returns a training and testing dataset of sudoku images as well as the dimension of a single image.
    @param split: The desired training to testing split size
    @return:
    """
    # Define the directory containing all the label directories
    main_directory = './cell_dataset'

    # Initialize lists to store image file paths and their corresponding labels
    file_paths = []
    labels = []

    # Iterate over each label directory
    for label in os.listdir(main_directory):
        label_directory = os.path.join(main_directory, label)
        if os.path.isdir(label_directory):
            # Iterate over each image file in the label directory
            for file in os.listdir(label_directory):
                if file.endswith('.jpg'):
                    # Append the file path and label to the respective lists
                    file_path = os.path.join(label_directory, file)
                    file_paths.append(file_path)
                    labels.append(int(label))

    labels = np.array(labels, dtype=np.int32)
    labels = torch.tensor(labels)
    all_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in file_paths]
    all_images = format_inputs_cells(all_images)

    if verbose:
        print(f"Shape labels: {labels.shape}, dtype: {labels.dtype}")
        print(f"Shape all_images: {all_images.shape}, dtype: {all_images.dtype}")

    dataset = TensorDataset(all_images, labels)
    train_size = int(split * len(labels))
    test_size = len(labels) - train_size

    train_data, test_data = random_split(dataset, [train_size, test_size])
    single_image_dimension = all_images.shape[1:]

    return train_data, test_data, single_image_dimension

def parseCells(warped: Union[str, np.ndarray], dat_path: str) -> Tuple[list, np.ndarray]:
    """
    Parses a warped image into 81 cells naively with their corresponding label
    @param warped:
    @param dat_path:
    @return:
    """
    if isinstance(warped, str):
        full_img = cv2.imread(warped, cv2.IMREAD_GRAYSCALE)
    elif isinstance(warped, np.ndarray):
        full_img = warped
    else:
        raise ValueError(f"Unsupported type {type(warped)}")
    # parse the image into a 9x9 grid and append the cells
    cells = []
    step_x = full_img.shape[1] // 9
    step_y = full_img.shape[0] // 9
    for m in range(9):
        for n in range(9):
            curr_cell = copy(full_img[m*step_y: (m+1)*step_y, n*step_x: (n+1)*step_x])
            cells.append(curr_cell)

    # parse the .dat file to get the corresponding labels for each cell in the image
    labels = parse_dat(dat_path).flatten()
    return cells, labels
