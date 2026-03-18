#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone OCR EN V5 Character Recognition Inference Script

This script provides a complete inference pipeline for PP-OCRv5 English character recognition.
It includes model loading, image preprocessing, inference, and post-processing.

Requirements:
    - PaddlePaddle (paddle)
    - OpenCV (cv2)
    - NumPy
    - PIL

Usage:
    # Basic usage
    python ocr_en_v5_inference_standalone.py --image_path ./demo.jpg --model_dir ./inference/en_PP-OCRv5_rec

    # With custom dictionary
    python ocr_en_v5_inference_standalone.py --image_path ./demo.jpg --model_dir ./model --dict_path ./custom_dict.txt

    # Batch processing
    python ocr_en_v5_inference_standalone.py --image_dir ./images/ --model_dir ./model

Model Download:
    Download EN PP-OCRv5 model from PaddleOCR model zoo:
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
"""

import os
import sys
import argparse
import cv2
import numpy as np
import math
from PIL import Image
import glob

# Import standalone post-processing
from rec_postprocess_standalone import CTCLabelDecode


class OCRENV5Recognizer:
    """
    Standalone OCR EN V5 Character Recognition Inference Class

    This class provides complete inference functionality for PP-OCRv5 English recognition models,
    including image preprocessing, model inference, and post-processing.

    Args:
        model_dir: Directory containing the inference model files
                   Supported formats:
                   - inference.pdmodel + inference.pdiparams (standard format)
                   - inference.json + inference.pdiparams (new format, PaddlePaddle 3.0+)
                   - model.pdmodel + model.pdiparams (alternative naming)
                   - model.json + model.pdiparams (alternative naming)
        dict_path: Path to character dictionary file
        use_gpu: Whether to use GPU for inference
        use_space_char: Whether to recognize space character
        rec_image_shape: Input image shape [C, H, W]
    """

    def __init__(
        self,
        model_dir,
        dict_path="./ppocr/utils/dict/ppocrv5_en_dict.txt",
        use_gpu=False,
        use_space_char=True,
        rec_image_shape=[3, 48, 320],
    ):
        self.rec_image_shape = rec_image_shape
        self.model_dir = model_dir

        # Initialize post-processor
        print(f"Loading character dictionary from: {dict_path}")
        self.postprocess_op = CTCLabelDecode(
            character_dict_path=dict_path,
            use_space_char=use_space_char,
        )
        print(f"Character set size: {len(self.postprocess_op.character)}")

        # Initialize predictor
        self.predictor = self._create_predictor(model_dir, use_gpu)
        print("Model loaded successfully!")

    def _create_predictor(self, model_dir, use_gpu=False):
        """
        Create PaddlePaddle inference predictor

        Args:
            model_dir: Directory containing inference model files
            use_gpu: Whether to use GPU

        Returns:
            Predictor object
        """
        try:
            import paddle.inference as paddle_infer
        except ImportError:
            print("Error: PaddlePaddle is not installed!")
            print("Please install it: pip install paddlepaddle-gpu or pip install paddlepaddle")
            sys.exit(1)

        # Find model files - support both "inference" and "model" prefixes
        # Also support both .pdmodel and .json formats
        params_file_path = None
        model_file_path = None
        file_name = None

        # Try to find .pdiparams file with different prefixes
        file_names = ["inference", "model"]
        for name in file_names:
            params_path = os.path.join(model_dir, f"{name}.pdiparams")
            if os.path.exists(params_path):
                params_file_path = params_path
                file_name = name
                break

        if params_file_path is None:
            # List available files to help user debug
            available_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
            raise FileNotFoundError(
                f"Model parameters file not found in: {model_dir}\n"
                f"Expected: 'inference.pdiparams' or 'model.pdiparams'\n"
                f"Available files: {available_files}\n"
                f"Please download the correct model files."
            )

        # Find model structure file (.json preferred over .pdmodel)
        json_path = os.path.join(model_dir, f"{file_name}.json")
        pdmodel_path = os.path.join(model_dir, f"{file_name}.pdmodel")

        if os.path.exists(json_path):
            model_file_path = json_path
            print(f"Found model structure: {file_name}.json (new format)")
        elif os.path.exists(pdmodel_path):
            model_file_path = pdmodel_path
            print(f"Found model structure: {file_name}.pdmodel")
        else:
            # List available files to help user debug
            available_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
            raise FileNotFoundError(
                f"Model structure file not found in: {model_dir}\n"
                f"Expected: '{file_name}.pdmodel' or '{file_name}.json'\n"
                f"Found parameters: {params_file_path}\n"
                f"Available files: {available_files}\n\n"
                f"Note: A complete model requires both structure (.pdmodel or .json) and parameters (.pdiparams) files.\n"
                f"Please ensure you have downloaded the complete model package."
            )

        print(f"Loading model from: {model_dir}")
        print(f"  Model structure: {os.path.basename(model_file_path)}")
        print(f"  Model parameters: {os.path.basename(params_file_path)}")

        # Configure predictor
        config = paddle_infer.Config(model_file_path, params_file_path)

        if use_gpu:
            config.enable_use_gpu(500, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(10)

        # Enable new IR and executor if available (required before memory optimization)
        if hasattr(config, "enable_new_ir"):
            config.enable_new_ir()
        if hasattr(config, "enable_new_executor"):
            config.enable_new_executor()

        # Memory optimization
        config.enable_memory_optim()
        config.disable_glog_info()

        # Configure passes and IR optimization
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # Create predictor
        predictor = paddle_infer.create_predictor(config)

        return predictor

    def resize_norm_img_svtr(self, img, image_shape):
        """
        Resize and normalize image for SVTR-based models (PP-OCRv5)

        This preprocessing is specific to SVTR architecture used in PP-OCRv5.
        Steps:
        1. Calculate aspect ratio and resize width
        2. Resize image maintaining aspect ratio
        3. Normalize to [-1, 1]
        4. Pad with zeros to target width

        Args:
            img: Input image (BGR format from OpenCV)
            image_shape: Target shape [C, H, W]

        Returns:
            Preprocessed image ready for inference
        """
        imgC, imgH, imgW = image_shape
        max_wh_ratio = imgW * 1.0 / imgH
        h, w = img.shape[0], img.shape[1]
        ratio = w * 1.0 / h
        max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
        imgW = int(imgH * max_wh_ratio)

        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        # Resize image
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")

        # Transpose to CHW format and normalize
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        # Pad to target width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def preprocess(self, img):
        """
        Preprocess image for recognition

        Args:
            img: Input image (numpy array in BGR format)

        Returns:
            Preprocessed image batch ready for model input
        """
        norm_img = self.resize_norm_img_svtr(img, self.rec_image_shape)
        norm_img = norm_img[np.newaxis, :]  # Add batch dimension
        return norm_img

    def predict(self, img_list):
        """
        Run inference on image list

        Args:
            img_list: List of images (numpy arrays in BGR format)

        Returns:
            List of (text, confidence) tuples
        """
        # Preprocess images
        batch_images = []
        for img in img_list:
            norm_img = self.preprocess(img)
            batch_images.append(norm_img)

        # Concatenate batch
        if len(batch_images) == 1:
            batch_input = batch_images[0]
        else:
            batch_input = np.concatenate(batch_images, axis=0)

        # Run inference
        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(batch_input)

        self.predictor.run()

        # Get output
        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])
        preds = output_tensor.copy_to_cpu()

        # Post-process
        results = self.postprocess_op(preds)

        return results

    def __call__(self, img_list):
        """
        Recognize text from images

        Args:
            img_list: List of images or single image

        Returns:
            List of recognition results
        """
        if not isinstance(img_list, list):
            img_list = [img_list]

        return self.predict(img_list)


def load_image(image_path):
    """
    Load image from file path

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return img


def get_image_list(image_path):
    """
    Get list of image paths

    Args:
        image_path: Path to image file or directory

    Returns:
        List of image file paths
    """
    if os.path.isfile(image_path):
        return [image_path]
    elif os.path.isdir(image_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(image_path, ext)))
        return sorted(image_list)
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='OCR EN V5 Character Recognition Inference')

    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to inference model directory')
    parser.add_argument('--dict_path', type=str,
                        default='./ppocr/utils/dict/ppocrv5_en_dict.txt',
                        help='Path to character dictionary')

    # Image arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to input image file or directory')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to input image directory (deprecated, use image_path)')

    # Inference arguments
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--use_space_char', type=bool, default=True,
                        help='Whether to recognize space character')
    parser.add_argument('--rec_image_shape', type=str, default='3,48,320',
                        help='Recognition image shape (C,H,W)')

    # Output arguments
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save recognition results')

    args = parser.parse_args()

    # Parse image shape
    rec_image_shape = [int(v) for v in args.rec_image_shape.split(',')]

    # Determine image path
    image_path = args.image_path or args.image_dir
    if image_path is None:
        print("Error: Please specify --image_path")
        sys.exit(1)

    # Initialize recognizer
    print("=" * 80)
    print("OCR EN V5 Character Recognition - Standalone Inference")
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")
    print(f"Dictionary path: {args.dict_path}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Image shape: {rec_image_shape}")
    print()

    recognizer = OCRENV5Recognizer(
        model_dir=args.model_dir,
        dict_path=args.dict_path,
        use_gpu=args.use_gpu,
        use_space_char=args.use_space_char,
        rec_image_shape=rec_image_shape,
    )

    # Get image list
    image_list = get_image_list(image_path)
    print(f"Found {len(image_list)} images to process")
    print()

    # Process images
    results_all = []
    for idx, img_path in enumerate(image_list):
        print(f"[{idx+1}/{len(image_list)}] Processing: {img_path}")

        # Load image
        img = load_image(img_path)
        print(f"  Image shape: {img.shape}")

        # Run recognition
        results = recognizer([img])

        # Display results
        for text, conf in results:
            print(f"  Result: '{text}'")
            print(f"  Confidence: {conf:.4f}")
            results_all.append({
                'image_path': img_path,
                'text': text,
                'confidence': conf
            })
        print()

    # Save results if specified
    if args.save_results:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            for result in results_all:
                f.write(f"{result['image_path']}\t{result['text']}\t{result['confidence']:.4f}\n")
        print(f"Results saved to: {args.save_results}")

    print("=" * 80)
    print(f"Total images processed: {len(image_list)}")
    print("=" * 80)


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Demo mode - show usage examples
        print("=" * 80)
        print("OCR EN V5 Character Recognition - Standalone Inference Script")
        print("=" * 80)
        print()
        print("This script provides standalone inference for PP-OCRv5 English recognition.")
        print()
        print("Usage Examples:")
        print("-" * 80)
        print()
        print("1. Single image recognition:")
        print("   python ocr_en_v5_inference_standalone.py \\")
        print("       --model_dir ./inference/en_PP-OCRv5_rec \\")
        print("       --image_path ./demo.jpg")
        print()
        print("2. Batch processing:")
        print("   python ocr_en_v5_inference_standalone.py \\")
        print("       --model_dir ./inference/en_PP-OCRv5_rec \\")
        print("       --image_path ./images/")
        print()
        print("3. Using GPU:")
        print("   python ocr_en_v5_inference_standalone.py \\")
        print("       --model_dir ./inference/en_PP-OCRv5_rec \\")
        print("       --image_path ./demo.jpg \\")
        print("       --use_gpu")
        print()
        print("4. Save results to file:")
        print("   python ocr_en_v5_inference_standalone.py \\")
        print("       --model_dir ./inference/en_PP-OCRv5_rec \\")
        print("       --image_path ./images/ \\")
        print("       --save_results ./results.txt")
        print()
        print("5. Custom dictionary:")
        print("   python ocr_en_v5_inference_standalone.py \\")
        print("       --model_dir ./inference/en_PP-OCRv5_rec \\")
        print("       --image_path ./demo.jpg \\")
        print("       --dict_path ./custom_dict.txt")
        print()
        print("-" * 80)
        print()
        print("Programmatic Usage:")
        print("-" * 80)
        print()
        print("from ocr_en_v5_inference_standalone import OCRENV5Recognizer")
        print("import cv2")
        print()
        print("# Initialize recognizer")
        print("recognizer = OCRENV5Recognizer(")
        print("    model_dir='./inference/en_PP-OCRv5_rec',")
        print("    dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',")
        print("    use_gpu=False")
        print(")")
        print()
        print("# Load image")
        print("img = cv2.imread('demo.jpg')")
        print()
        print("# Run recognition")
        print("results = recognizer([img])")
        print()
        print("# Get results")
        print("for text, conf in results:")
        print("    print(f'Text: {text}, Confidence: {conf:.4f}')")
        print()
        print("-" * 80)
        print()
        print("Model Download:")
        print("-" * 80)
        print("Download EN PP-OCRv5 models from:")
        print("https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md")
        print()
        print("=" * 80)
