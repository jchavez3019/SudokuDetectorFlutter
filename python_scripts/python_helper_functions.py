from matplotlib import pyplot as plt
import numpy as np
import cv2
from copy import copy
import os
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data.dataset import Subset
from torch import Tensor, Size
import torch.optim as optim
from pathlib import Path
from functools import partial
from typing import *

class SudokuDataset(torch.utils.data.Dataset):
    """
    Dataset class for the Sudoku dataset. The occurrences of each label are counted,
    and the inverse frequency is calculated to address class imbalance.
    In particular, the dataset will typically contain a lot of empty cells and therefore
    samples with a label of zero will dominate the dataset.
    """
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
        self.x = images.to(device='cpu')
        self.y = labels.to(device='cpu')
        self.total_size = labels.numel() # total size of the dataset
        # First, compute the inverse frequency which will give larger weights to more
        # rare occurrences. Then, normalize and scale by the number of classes.
        label_frequency, _ = torch.histogram(labels.to(dtype=torch.float32), bins=num_classes)
        self.weights = 1.0 / label_frequency
        self.weights = (self.weights / self.weights.sum()) * num_classes

    def __len__(self)->int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

def get_sudoku_dataset(dataset_path: str, split: float = 0.8,
                       supported_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                       verbose: bool = False) -> Tuple[torch.utils.data.dataset, Subset, Subset, Size]:
    """
    Returns a training and testing dataset of sudoku images as well as the dimension of a single image.
    :param dataset_path:            Path to the sudoku dataset.
    :param split:                   The desired training to testing split ratio.
    :param supported_extensions:    The type of images supported by the dataset.
    :param verbose:                 If true, prints additional information.
    :return:
    """
    # Initialize lists to store image file paths and their corresponding labels
    file_paths = []
    labels = []

    for label_path in os.listdir(dataset_path):
        # Iterate over each label directory
        label_directory = os.path.join(dataset_path, label_path)
        if not os.path.isdir(label_directory):
            # the label_directory is not a valid path
            if verbose:
                print(f"Skipping {label_path}")
            continue
        for ext in supported_extensions:
            # Iterate through each supported file extension and add them to the total
            # set of labels and file paths

            # this returns all file paths in the label directory that match the current extension
            ext_paths = [str(f) for f in Path(label_directory).glob(f"*{ext}")]
            # extend the labels for the current set of images
            labels.extend([int(label_path)] * len(ext_paths))
            # extend the image file paths
            file_paths.extend(ext_paths)

    # convert labels from a list to a tensor
    labels = torch.as_tensor(labels, dtype=torch.int32)
    # load in and decode all the image paths
    all_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in file_paths]
    # convert each decoded image into a tensor
    all_images = format_inputs_cells(all_images)

    if verbose:
        print("Dataset shapes:")
        print(f"\tShape labels: {labels.shape}, dtype: {labels.dtype}")
        print(f"\tShape all_images: {all_images.shape}, dtype: {all_images.dtype}")

    # compile all images and labels into a TensorDataset and compute the sizes
    dataset = SudokuDataset(all_images, labels)
    train_size = int(split * len(labels))
    test_size = len(labels) - train_size

    # perform a randomized train/test split
    train_data, test_data = random_split(dataset, [train_size, test_size])
    single_image_dimension = all_images.shape[1:]

    return dataset, train_data, test_data, single_image_dimension

def resize_image_high_quality(image: Union[cv2.Mat, np.ndarray[Any, np.dtype]], target_width: int = 3400, target_height: int = 2500,
                              interpolation_method: int = cv2.INTER_LANCZOS4, verbose: bool = False):
    """
    Resize the image to the target dimension only if current dimensions are smaller. Uses high-quality interpolation methods.
    :param image:                   Input image (numpy array)
    :param target_width:            Target width (default: 3400)
    :param target_height:           Target height (default: 2500)
    :param interpolation_method:    INTER_CUBIC produces high quality scaling, and INTER_LANCZOS4 has even higher
                                    quality with slower speed.
    :param verbose:                 If true, prints additional information.
    :return:                        Resized image or original image if already large enough.
    """
    current_height, current_width = image.shape[:2]
    if current_width >= target_width and current_height >= target_height:
        # if the image is already larger than the target size, return it as is
        if verbose: print(f"Image already large enough ({current_width}x{current_height}). No resize needed.")
        return image

    if verbose: print(f"Resizing from {current_width}x{current_height} to {target_width}x{target_height}")

    # resize the image
    resized = cv2.resize(image, (target_width, target_height), interpolation=interpolation_method)

    return resized

def extract_sudoku_puzzle(image_path: str, new_size: Optional[Tuple[int, int]] = None, verbose: bool = False,
                          display: bool = False) -> Union[cv2.Mat, np.ndarray[Any, np.dtype]]:
    """
    Extracts a warped and cropped sudoku puzzle from an image.
    :param image_path:  The path of the image to extract the sudoku puzzle from.
    :param new_size:    If specified, the size of the final image should match.
    :param verbose:     If true, prints additional information.
    :param display:     If true, plots are displayed of each step of transformation.
    :return:            A resulting image that extracts soley the Sudoku puzzle from the original image.
    """
    original_image = resize_image_high_quality(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR),
                                                            cv2.COLOR_BGR2RGB))
    colored_image = copy(original_image)
    gray_image = resize_image_high_quality(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

    img_width = gray_image.shape[1]
    img_height = gray_image.shape[0]

    if verbose:
        print(f"Shape of image: {img_width}, {img_height}")

    modified_image = cv2.addWeighted(gray_image, 1.2, gray_image, 0, 0.0)
    modified_image = cv2.medianBlur(modified_image, 25)
    modified_image = cv2.adaptiveThreshold(modified_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    modified_image = cv2.Canny(modified_image, 250, 200)
    modified_image = cv2.dilate(modified_image, np.ones((11, 11), np.uint8), np.zeros(gray_image.shape), [-1, 1])

    contours, _ = cv2.findContours(modified_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest contour
    contour_areas = np.array([cv2.contourArea(cont) for cont in contours])
    area_indices = np.flip(np.argsort(contour_areas))

    # Just to analyze the area of the largest contour
    largest_area = contour_areas[area_indices[0]]
    total_area = img_width * img_height

    # filter out contours whose area is nearly the entire image which would indicate that there is a border around
    # the image
    l_i = 0  # choose the first contour below the 98% threshold
    area_threshold = 0.98
    for i, area_idx in enumerate(area_indices):
        curr_area = contour_areas[area_idx]
        if curr_area / total_area < area_threshold:
            l_i = i
            break
    area_indices = area_indices[l_i:]

    # for a_idx in area_indices:
    # perimeter = cv2.arcLength(contours[a_idx[0], True) * 0.02
    # approx = cv2.approxPolyDP(contours[a_idx[0], perimeter, True)
    approx = cv2.approxPolyN(contours[area_indices[0]], nsides=4, epsilon_percentage=-1.0, ensure_convex=True)

    contourArea = cv2.contourArea(approx)
    # if len(approx) == 4 and abs(contourArea) > 2000 and cv2.isContourConvex(approx):
    if verbose:
        print(f"This is a valid contour; Area of contour is {contourArea}; Shape of approx is {approx.shape}")
    for i in range(len(approx)):
        point = np.squeeze(approx[i])
        print(f"Point_{i}: ({point[0]},{point[1]})")

    temp_approx = np.squeeze(approx)
    if verbose:
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
    if verbose:
        print(f"Reordered: \n{reordered_approx}")
    approx = reordered_approx
    # break

    if img_height < img_width:
        orientation = 'landscape'
        square_size = int(img_height * 0.90)
    else:
        orientation = 'portrait'
        square_size = int(img_width * 0.90)

    # get the perspective transform matrix
    target_coordinates = np.array([[0, 0], [0, square_size], [square_size, square_size], [square_size, 0]],
                                  dtype=np.float32)
    # target_coordinates = np.array([[square_size, 0], [square_size, square_size], [0, square_size], [0, 0]],
    #                               dtype=np.float32)
    transformation_matrix = cv2.getPerspectiveTransform(np.squeeze(approx).astype(np.float32), target_coordinates)

    # Apply the perspective transformation
    result_img = cv2.warpPerspective(colored_image, transformation_matrix, (int(square_size), int(square_size)))

    if verbose:
        print(f"Orientation: {orientation}, Square Size: {square_size}, Image dimensions: {original_image.shape}")
        print(f"Target (x,y) coordinates:\n{target_coordinates}")

    if display:
        # display results
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1), plt.imshow(colored_image, 'gray')
        plt.title('Original Image')
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(result_img, 'gray')
        plt.title('Warped Image')
        plt.xticks([]), plt.yticks([])
        plt.show()

    if new_size is not None:
        result_img = cv2.resize(result_img, new_size)

    return result_img

def format_inputs_cells(cells: list[np.ndarray]) -> Tensor:
    """
    Given a list of, presumably 81, decoded cells that the Sudoku puzzle is composed of, convert them to
    Tensors that is compatible with a Torch model.
    :param cells:   List of decoded cells.
    :return:        Tensor format of the cells.
    """
    # resize each image to be 50x50
    cells = np.array(
        [cv2.resize(cell, (50, 50)).astype(np.float32) / 255.0 for cell in
         cells])
    # convert all images to tensors and add a channel dimension
    all_images = torch.tensor(cells)
    all_images = all_images.unsqueeze(1)  # channel dimension indicates to the NN the images are black and white

    return all_images

def parse_processed_dataset(dataset_path: str, out_path: str,
                            supported_extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),):
    """
    Creates a number dataset from a **processed** dataset. Processed in the sense that each image from the dataset
    solely contains a bird's eye view of the Sudoku puzzle and nothing else.
    :param dataset_path:            Path to a **processed** dataset.
    :param out_path:                Path to save parsed cells in the processed dataset.
    :param supported_extensions:    Images supported by the dataset.
    """
    # get the names of all files in the dataset_path
    all_file_names = os.listdir(dataset_path)
    # convert to Path objects
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)
    # get all the filenames of all the supported images
    image_file_names = sorted(
        [dataset_path / filename for filename in all_file_names if filename.endswith(supported_extensions)]
    )
    # get the .dat file for all the images that were extracted
    dat_filenames = sorted(
        [filename.with_suffix('.dat') for filename in image_file_names]
    )
    # this counter keeps track of the image count for each label
    folder_counter = np.zeros(10, dtype=np.int32)
    for image_fn, dat_fn in zip(image_file_names, dat_filenames):
        # parse the image to get the cells and corresponding labels
        cells, labels = parse_cells(str(image_fn), str(dat_fn))

        for i, (label, cell) in enumerate(zip(labels, cells)):
            # enumerate over each label and the cell's image
            label_path = out_path / f"{label}"
            if not os.path.exists(label_path):
                # create the path to the label's folder if it does not exist yet
                os.makedirs(label_path)

            try:
                # save the cell's image to the appropriate path and increment the label counter
                cv2.imwrite(str(label_path / f"{folder_counter[label]}.jpg"), cell)
                folder_counter[label] += 1
            except IndexError:
                print(f"Len cells {len(cells)}, len labels: {len(labels)}")
                print(f"Index out of range: {label}. File name: {image_fn}")


def parse_cells(extracted_img: Union[str, np.ndarray], dat_path: str) -> Tuple[list[np.ndarray], np.ndarray]:
    """
    Parses a processed image that solely contains a Sudoku puzzle into 81 cells naively with their corresponding label.
    :param extracted_img:   An image that contains solely the Sudoku puzzle, potentially extracted from a different
                            image.
    :param dat_path:        Path to the puzzle's labels.
    :return:                List of 81 cells and corresponding labels.
    """
    if isinstance(extracted_img, str):
        full_img = cv2.imread(extracted_img, cv2.IMREAD_GRAYSCALE)
    elif isinstance(extracted_img, np.ndarray):
        full_img = extracted_img
    else:
        raise ValueError(f"Unsupported type {type(extracted_img)}")
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

def parse_dat(dat_path: str) -> np.ndarray:
    """
    Parses a .dat file and returns the labels of a sudoku puzzle as a 1D np array
    :param dat_path:
    :return:
    """
    numbers = [] # a 2D array of the number in each cell
    with open(dat_path, 'r') as file:
        for i, line in enumerate(file):
            if i < 2:
                # the first two lines specify attributes of the associated image
                continue
            # Split the string into substrings based on spaces
            number_strings = line.split()
            # Convert each substring to an integer and store them in a list
            numbers.append([int(num_str) for num_str in number_strings])

    return np.array(numbers)

class CannyEdgeDetector:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_blur(img: cv2.Mat, ksize=(5, 5), sigma: float=1.0)->cv2.Mat:
        """
        Applies a Gaussian Blur filter to the image.
        :param img:     Image to blur.
        :param ksize:   Size of the Gaussian kernel.
        :param sigma:   Variance of the Gaussian kernel.
        :return:        Blurred image.
        """
        return cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def compute_gradients(img: cv2.Mat) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the Sobel filter to calculate the magnitude and angle of the gradients in an image.
        :param img: Image to process.
        :return:    Magnitude and angle of the gradients.
        """
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        return mag, angle

    @staticmethod
    def non_maximum_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """
        Retain only the local maxima in the gradient direction.
        :param mag:     The gradient magnitudes.
        :param angle:   The gradient angles.
        :return:        The gradient magnitudes' local maxima.
        """
        H, W = mag.shape
        output = np.zeros((H, W), dtype=np.float64)

        angle = angle % 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q, r = 255, 255
                a = angle[i, j]

                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                elif 22.5 <= a < 67.5:
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                elif 67.5 <= a < 112.5:
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                elif 112.5 <= a < 157.5:
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]

                if mag[i, j] >= q and mag[i, j] >= r:
                    output[i, j] = mag[i, j]
        return output

    @staticmethod
    def double_threshold(img: cv2.Mat, low_thresh: float, high_thresh: float) -> Tuple[np.ndarray, int, int]:
        """
        We classify the intensity of the pixels in the image to 3 classes based on the specified thresholds:

        1) The pixel has large intensity
        2) The pixel has small intensity
        3) The pixel has no intensity

        :param img:         The image to threshold.
        :param low_thresh:  Low threshold for continuing an edge.
        :param high_thresh: High threshold for beginning an edge.
        :return:            The intensity of each pixel intensity values.
        """
        strong = 255 # intensity for strong edges
        weak = 75 # intensity for weak edges

        # map the pixel indices to 1) strong edge 2) weak edge 3) no edge
        strong_i, strong_j = np.where(img >= high_thresh)
        weak_i, weak_j = np.where((img <= high_thresh) & (img >= low_thresh))

        # color in the output
        output = np.zeros_like(img, dtype=np.uint8)
        output[strong_i, strong_j] = strong
        output[weak_i, weak_j] = weak

        return output, weak, strong

    @staticmethod
    def hysteresis(img: cv2.Mat, weak:int=75, strong:int=255) -> cv2.Mat:
        """
        Process all the pixels with weak intensity, retaining only those that are part of a continued edge.
        :param img:     Image to apply hysteresis.
        :param weak:    The weak intensity value.
        :param strong:  The strong intensity value.
        :return:        Image with filtered edges.
        """
        H, W = img.shape

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img[i, j] == weak:
                    # we process all the pixels with weak intensity

                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        # if the pixel has a neighbor with strong intensity, we elevate the intensity of this pixel
                        img[i, j] = strong
                    else:
                        # demote the pixel's intensity
                        img[i, j] = 0

        return img

    def __call__(self, gray: cv2.Mat,
                 low_thresh:float=50, high_thresh:float=150) -> Tuple[cv2.Mat, np.ndarray, np.ndarray]:
        """

        :param gray:
        :param low_thresh:
        :param high_thresh:
        :return:
        """

        blur = CannyEdgeDetector.gaussian_blur(gray, (5, 5), 1.4)
        mag, angle = CannyEdgeDetector.compute_gradients(blur)
        nms = CannyEdgeDetector.non_maximum_suppression(mag, angle)
        dt, weak, strong = CannyEdgeDetector.double_threshold(nms, low_thresh, high_thresh)
        result = CannyEdgeDetector.hysteresis(dt, weak, strong)

        return result, mag, angle

### Custom LR Schedulers
def linear_decay(epoch, initial_lr, final_lr, total_epochs, last_decay:float=1e-3):
    """

    :param epoch:
    :param initial_lr:
    :param final_lr:
    :param total_epochs:
    :param last_decay:
    :return:
    """
    # # FIXME: Currently hard-coded to a tailored configuration that shows stable convergence. The parameters should
    # # be modified instead of hard-coded.
    # if epoch < 25:
    #     total_epochs = 25
    #     return 1 - epoch / total_epochs * (1 - final_lr / initial_lr)
    # else:
    #     return last_decay
    return 1 - epoch / total_epochs * (1 - final_lr / initial_lr)

def get_lr_scheduler(scheduler_type: str, optimizer: optim.Optimizer, lr_parameters: Dict[str, Any],
                     verbose: bool = False):
    """
    Return the learning rate scheduler to use from a common set.
    :param scheduler_type:
    :param optimizer:
    :param lr_parameters:
    :param verbose:
    :return:
    """
    if verbose:
        print(f"Using {scheduler_type} learning rate scheduler with parameters: {lr_parameters}")

    # set linear LR scheduler
    scheduler = None
    if scheduler_type == 'step':
        step_size = lr_parameters.pop('step_size')
        gamma = lr_parameters.pop('gamma')
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                   gamma=gamma, **lr_parameters)
    elif scheduler_type == 'linear':
        decay_func = partial(linear_decay, **lr_parameters)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_func)
    elif scheduler_type == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **lr_parameters)
    elif scheduler_type == 'none':
        pass
    else:
        raise ValueError(f"Scheduler type of {scheduler_type} is not recognized")

    return scheduler