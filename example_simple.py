#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Example: OCR EN V5 Character Recognition

This is a minimal example showing how to use the standalone scripts
for character recognition.
"""

# Example 1: Using the complete inference pipeline
print("=" * 60)
print("Example 1: Complete Inference Pipeline")
print("=" * 60)
print()
print("from ocr_en_v5_inference_standalone import OCRENV5Recognizer")
print("import cv2")
print()
print("# Initialize recognizer")
print("recognizer = OCRENV5Recognizer(")
print("    model_dir='./en_PP-OCRv5_mobile_rec_infer',")
print("    dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',")
print("    use_gpu=False")
print(")")
print()
print("# Load and recognize image")
print("img = cv2.imread('text_image.jpg')")
print("results = recognizer([img])")
print()
print("# Print results")
print("for text, confidence in results:")
print("    print(f'Text: {text}')")
print("    print(f'Confidence: {confidence:.4f}')")
print()
print()

# Example 2: Using only post-processing
print("=" * 60)
print("Example 2: Post-processing Only")
print("=" * 60)
print()
print("from rec_postprocess_standalone import CTCLabelDecode")
print("import numpy as np")
print()
print("# Initialize post-processor")
print("postprocessor = CTCLabelDecode(")
print("    character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',")
print("    use_space_char=True")
print(")")
print()
print("# Assume you have model predictions")
print("# Shape: [batch_size, time_steps, num_classes]")
print("model_predictions = get_model_predictions()  # Your model")
print()
print("# Apply post-processing")
print("results = postprocessor(model_predictions)")
print()
print("# Get recognized text")
print("for text, confidence in results:")
print("    print(f'{text} (conf: {confidence:.3f})')")
print()
print()

# Example 3: Batch processing
print("=" * 60)
print("Example 3: Batch Processing Multiple Images")
print("=" * 60)
print()
print("from ocr_en_v5_inference_standalone import OCRENV5Recognizer")
print("import cv2")
print("import glob")
print()
print("# Initialize recognizer")
print("recognizer = OCRENV5Recognizer(")
print("    model_dir='./en_PP-OCRv5_mobile_rec_infer'")
print(")")
print()
print("# Load multiple images")
print("image_paths = glob.glob('./images/*.jpg')")
print("images = [cv2.imread(path) for path in image_paths]")
print()
print("# Process batch")
print("results = recognizer(images)")
print()
print("# Save results")
print("with open('results.txt', 'w') as f:")
print("    for path, (text, conf) in zip(image_paths, results):")
print("        f.write(f'{path}\\t{text}\\t{conf:.4f}\\n')")
print()
print()

# Actual demonstration
print("=" * 60)
print("Live Demo: Post-processing")
print("=" * 60)
print()

from rec_postprocess_standalone import CTCLabelDecode
import numpy as np
import os

# Check if dictionary exists
dict_path = './ppocr/utils/dict/ppocrv5_en_dict.txt'
if os.path.exists(dict_path):
    print(f"✓ Using dictionary: {dict_path}")
    postprocessor = CTCLabelDecode(
        character_dict_path=dict_path,
        use_space_char=True
    )
else:
    print("⚠ Dictionary not found, using default")
    postprocessor = CTCLabelDecode()

print(f"✓ Character set size: {len(postprocessor.character)}")
print()

# Create a simple prediction for "HELLO"
print("Creating prediction for: 'HELLO'")
num_classes = len(postprocessor.character)
time_steps = 15

# Build prediction
preds = np.zeros((1, time_steps, num_classes))
char_to_idx = postprocessor.dict

sequence = [
    0,  # blank
    char_to_idx.get('H', 0),
    0,  # blank
    char_to_idx.get('E', 0),
    0,  # blank
    char_to_idx.get('L', 0),
    char_to_idx.get('L', 0),
    0,  # blank
    char_to_idx.get('O', 0),
    0,  # blank
]

# Fill remaining with blanks
sequence.extend([0] * (time_steps - len(sequence)))

# Set probabilities
for t, idx in enumerate(sequence):
    preds[0, t, idx] = 0.90
    preds[0, t, :] += np.random.rand(num_classes) * 0.05

# Normalize
preds = preds / preds.sum(axis=2, keepdims=True)

# Decode
results = postprocessor(preds)
text, confidence = results[0]

print(f"✓ Decoded text: '{text}'")
print(f"✓ Confidence: {confidence:.4f}")
print()
print("=" * 60)
