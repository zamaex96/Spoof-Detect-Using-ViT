import cv2
import numpy as np
import os
import random
from rembg import remove
from PIL import Image
import onnxruntime as ort

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gauss)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def remove_background(image):
    """
    Removes the background from an image using the rembg package.
    """
    # Convert the OpenCV BGR image to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Remove background using rembg
    output = remove(pil_image)

    # Convert output (PIL Image) back to a NumPy array and to BGR color space
    output_np = np.array(output)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    return output_bgr


def random_transform(image):
    """
    Applies a series of random transformations to the input image.
    """
    # Random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Random brightness adjustment (scaling pixel values)
    brightness_factor = random.uniform(0.8, 1.2)
    bright_adjusted = cv2.convertScaleAbs(rotated, alpha=brightness_factor, beta=0)

    # Add Gaussian noise
    transformed = add_gaussian_noise(bright_adjusted)

    return transformed


def generate_spoof_images(input_image_path, output_folder, num_images=100):
    """
    Generates spoof images by first removing the background and then
    applying random transformations to the input image.
    """
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the real portrait image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_image_path}")

    # Remove the background
    image_no_bg = remove_background(image)

    # Generate and save spoof images
    for i in range(num_images):
        spoof_image = random_transform(image_no_bg)
        output_path = os.path.join(output_folder, f"spoof_{i + 1:03d}.jpg")
        cv2.imwrite(output_path, spoof_image)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    input_image_path = r"C:\SpoofDetect\dataset\real\real1.jpg"  # Path to your real portrait image
    output_folder = r"C:\SpoofDetect\dataset\spoofed"  # Output folder for spoof images
    generate_spoof_images(input_image_path, output_folder, num_images=10)
