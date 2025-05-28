"""
Interactive Sudoku Dataset Processor

This script processes images from the original dataset through a preprocessing pipeline
and allows manual quality control filtering. High-quality warped images are saved to
the warped_dataset directory.

Usage:
    python process_sudoku_dataset.py [--input INPUT_DIR] [--output OUTPUT_DIR]

Controls:
    y/Y - Accept and save the processed image
    n/N - Reject the processed image
    s/S - Skip this image (continue without deciding)
    q/Q - Quit the program
    r/R - Restart processing from current image
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import shutil
from tqdm import tqdm

from python_helper_functions import detectSudokuPuzzle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/sudoku_dataset_warped_filter_improved.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SudokuDatasetProcessor:
    """Main processor class for handling sudoku dataset processing with user interaction."""

    def __init__(self, input_dir: Path, output_dir: Path, skip_existing_outputs: bool,
                 supported_extensions: List[str] = None):
        """

        :param input_dir:
        :param output_dir:
        :param skip_existing_outputs:
        :param supported_extensions:
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.skip_existing_outputs = skip_existing_outputs
        self.supported_extensions = supported_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'processed': 0,
            'accepted': 0,
            'rejected': 0,
            'skipped': 0,
            'errors': 0
        }

    def get_image_files(self) -> List[Path]:
        """Get all supported image files from input directory."""
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            # image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def get_user_decision(self, image_path: Path, processed_image) -> str:
        """
        Display processed image and get user decision.
        :param image_path:  Path to the original image
        :return:            'y', 'n', 's' (skip), 'q' (quit), or 'r' (restart)
        """
        # Display the processed image
        if processed_image is not None:
            # Resize for display if too large
            # display_img = self._resize_for_display(processed_image)
            display_img = processed_image
            window_name = f"Processed: {image_path.name} - (y)es/(n)o/(s)kip/(q)uit/(r)estart"
            cv2.imshow(window_name, display_img)

            print(f"\nProcessing: {image_path.name}")
            print("Controls: (y)es - accept, (n)o - reject, (s)kip, (q)uit, (r)estart")

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key in [ord('y'), ord('Y')]:
                    cv2.destroyAllWindows()
                    return 'y'
                elif key in [ord('n'), ord('N')]:
                    cv2.destroyAllWindows()
                    return 'n'
                elif key in [ord('s'), ord('S')]:
                    cv2.destroyAllWindows()
                    return 's'
                elif key in [ord('q'), ord('Q')]:
                    cv2.destroyAllWindows()
                    return 'q'
                elif key in [ord('r'), ord('R')]:
                    cv2.destroyAllWindows()
                    return 'r'
                elif key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    return 'q'
                else:
                    print("Invalid key. Use: y(es), n(o), s(kip), q(uit), r(estart)")
        else:
            print(f"Failed to process {image_path.name}")
            print("(s)kip or (q)uit?")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in [ord('s'), ord('S')]:
                    return 's'
                elif key in [ord('q'), ord('Q')]:
                    return 'q'

    def _resize_for_display(self, image, max_width: int = 1200, max_height: int = 900):
        """Resize image for display if it's too large."""
        h, w = image.shape[:2]
        if w <= max_width and h <= max_height:
            return image

        # Calculate scaling factor
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def process_single_image(self, image_path: Path) -> Optional[any]:
        """Process a single image using the sudoku detection pipeline."""
        try:
            result = detectSudokuPuzzle(str(image_path), new_size=(50*9, 50*9), display=False)
            return result

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            return None

    def copy_associated_files(self, source_path: Path, dest_path: Path) -> bool:
        """
        Copy associated files (like .dat files) along with the image.
        :param source_path: Path to the original image.
        :param dest_path:   Path where the processed image and its labels will be saved to.
        :return:
        """
        try:
            # Copy .dat file if it exists
            dat_source = source_path.with_suffix('.dat')
            if dat_source.exists():
                dat_dest = dest_path.with_suffix('.dat')
                shutil.copy2(dat_source, dat_dest)
                logger.info(f"Copied associated file: {dat_source.name}")
            else:
                logger.warning(f"Associated {dat_source.name} not found")
            return True
        except Exception as e:
            logger.error(f"Error copying associated files: {str(e)}")
            return False

    def save_processed_image(self, processed_image, output_path: Path, source_path: Path) -> bool:
        """
        Save the processed image and copy associated files.
        :param processed_image: The warped image using our preprocessing pipeline.
        :param output_path:     The path to save the image to.
        :param source_path:     The path to the original image.
        :return:
        """
        try:
            # Save the processed image
            success = cv2.imwrite(str(output_path), processed_image)
            if not success:
                logger.error(f"Failed to save image: {output_path}")
                return False

            # Copy associated files
            self.copy_associated_files(source_path, output_path)

            logger.info(f"Saved: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            return False

    def print_progress(self, current: int, total: int):
        """Print current progress and statistics."""
        progress = (current / total) * 100 if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"Progress: {current}/{total} ({progress:.1f}%)")
        print(f"Accepted: {self.stats['accepted']}, Rejected: {self.stats['rejected']}")
        print(f"Skipped: {self.stats['skipped']}, Errors: {self.stats['errors']}")
        print(f"{'='*60}")

    def process_dataset(self, resume_from: int = 0) -> bool:
        """Process the entire dataset with user interaction."""
        image_files = self.get_image_files()

        if not image_files:
            logger.warning(f"No supported image files found in {self.input_dir}")
            return False

        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        try:
            for i in range(resume_from, len(image_files)):
                image_path = image_files[i]
                output_path = self.output_dir / image_path.name

                self.print_progress(i, len(image_files))

                if self.skip_existing_outputs and output_path.exists():
                    # Skip if output already exists
                    logger.info(f"Output already exists, skipping: {image_path.name}")
                    continue

                # Process the image
                logger.info(f"Processing: {image_path.name}")
                processed_image = self.process_single_image(image_path)
                self.stats['processed'] += 1

                if processed_image is None:
                    self.stats['errors'] += 1
                    continue

                # Get user decision
                decision = self.get_user_decision(image_path, processed_image)

                if decision == 'y':
                    if self.save_processed_image(processed_image, output_path, image_path):
                        self.stats['accepted'] += 1
                        logger.info(f"Accepted: {image_path.name}")
                    else:
                        self.stats['errors'] += 1

                elif decision == 'n':
                    self.stats['rejected'] += 1
                    logger.info(f"Rejected: {image_path.name}")

                elif decision == 's':
                    self.stats['skipped'] += 1
                    logger.info(f"Skipped: {image_path.name}")

                elif decision == 'q':
                    logger.info("User requested quit")
                    break

                elif decision == 'r':
                    logger.info("Restarting current image")
                    i -= 1  # This will be incremented by the loop
                    continue

        except KeyboardInterrupt:
            logger.info("Process interrupted by user")

        finally:
            cv2.destroyAllWindows()
            self.print_final_stats()

        return True

    def print_final_stats(self):
        """Print final processing statistics."""
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total processed: {self.stats['processed']}")
        print(f"Accepted: {self.stats['accepted']}")
        print(f"Rejected: {self.stats['rejected']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")

        if self.stats['processed'] > 0:
            acceptance_rate = (self.stats['accepted'] / self.stats['processed']) * 100
            print(f"Acceptance rate: {acceptance_rate:.1f}%")
        print(f"{'='*60}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process sudoku dataset with manual quality control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='./original_dataset',
        help='Input directory containing original images (default: original_dataset)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./warped_dataset',
        help='Output directory for processed images (default: warped_dataset)'
    )

    parser.add_argument(
        '--resume', '-r',
        type=int,
        default=0,
        help='Resume processing from image index (default: 0)'
    )

    parser.add_argument(
        "--skip_existing", "-s",
        action="store_true",
        help="Skip images that already exist in the output directory."
    )

    args = parser.parse_args()

    # Create processor and run
    processor = SudokuDatasetProcessor(
        input_dir=args.input,
        output_dir=args.output,
        skip_existing_outputs=args.skip_existing
    )

    try:
        success = processor.process_dataset(resume_from=args.resume)
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())