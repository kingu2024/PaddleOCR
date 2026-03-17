#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Script for OCR EN V5 Character Recognition

This demo shows how to use the standalone inference and post-processing scripts.
It includes examples for both complete inference and post-processing only usage.

Usage:
    python demo_ocr_en_v5.py
"""

import numpy as np
import sys
import os

# Import the standalone modules
from rec_postprocess_standalone import CTCLabelDecode

print("=" * 80)
print("OCR EN V5 Character Recognition - Demo")
print("=" * 80)
print()

# ============================================================================
# Demo 1: Post-processing Only
# ============================================================================
print("[Demo 1] Post-processing Only - Standalone CTCLabelDecode")
print("-" * 80)

# Initialize post-processor
dict_path = "./ppocr/utils/dict/ppocrv5_en_dict.txt"
if os.path.exists(dict_path):
    postprocessor = CTCLabelDecode(
        character_dict_path=dict_path,
        use_space_char=True
    )
    print(f"✓ Loaded character dictionary: {dict_path}")
    print(f"✓ Character set size: {len(postprocessor.character)}")
else:
    print(f"⚠ Character dictionary not found at {dict_path}")
    print("  Using default character set (0-9, a-z)")
    postprocessor = CTCLabelDecode(use_space_char=False)
    print(f"✓ Character set size: {len(postprocessor.character)}")

print()

# Simulate model output for demonstration
batch_size = 2
time_steps = 25
num_classes = len(postprocessor.character)

print(f"Simulating CTC model output:")
print(f"  - Batch size: {batch_size}")
print(f"  - Time steps: {time_steps}")
print(f"  - Number of classes: {num_classes}")
print()

# Create simulated predictions
np.random.seed(123)
simulated_preds = np.random.rand(batch_size, time_steps, num_classes)

# Apply post-processing
results = postprocessor(simulated_preds)

print("Post-processing Results:")
for idx, (text, confidence) in enumerate(results):
    print(f"  Sample {idx + 1}:")
    print(f"    Text: '{text}'")
    print(f"    Confidence: {confidence:.4f}")

print()
print()

# ============================================================================
# Demo 2: Create Realistic CTC Predictions
# ============================================================================
print("[Demo 2] Realistic CTC Predictions for Known Text")
print("-" * 80)

# Define target texts to encode
target_texts = ["HELLO", "OCR", "PADDLE"]

char_to_idx = postprocessor.dict
blank_idx = 0

print("Creating predictions for target texts:")
for target_text in target_texts:
    print(f"\nTarget: '{target_text}'")

    # Create prediction sequence with blanks
    time_steps = len(target_text) * 3 + 5  # Extra space for blanks
    preds = np.zeros((1, time_steps, num_classes))

    # Build sequence: blank, char, blank, char, ...
    sequence = [blank_idx]
    for char in target_text:
        char_idx = char_to_idx.get(char, blank_idx)
        sequence.extend([char_idx, blank_idx])

    # Pad with blanks
    while len(sequence) < time_steps:
        sequence.append(blank_idx)
    sequence = sequence[:time_steps]

    # Set high probabilities for target characters
    for t, char_idx in enumerate(sequence):
        preds[0, t, char_idx] = 0.90
        # Add small noise to other classes
        preds[0, t, :] += np.random.rand(num_classes) * 0.05

    # Normalize
    preds = preds / preds.sum(axis=2, keepdims=True)

    # Decode
    results = postprocessor(preds)
    decoded_text, conf = results[0]

    match_symbol = "✓" if decoded_text == target_text else "✗"
    print(f"  Decoded: '{decoded_text}' {match_symbol}")
    print(f"  Confidence: {conf:.4f}")

print()
print()

# ============================================================================
# Demo 3: Complete Inference (if model available)
# ============================================================================
print("[Demo 3] Complete Inference with Model")
print("-" * 80)

model_dir = "./en_PP-OCRv5_mobile_rec_infer"
if os.path.exists(model_dir):
    try:
        from ocr_en_v5_inference_standalone import OCRENV5Recognizer
        import cv2

        print(f"✓ Model directory found: {model_dir}")
        print("Initializing recognizer...")

        recognizer = OCRENV5Recognizer(
            model_dir=model_dir,
            dict_path=dict_path if os.path.exists(dict_path) else None,
            use_gpu=False
        )

        print("✓ Recognizer initialized successfully!")
        print()
        print("You can now use the recognizer like this:")
        print()
        print("  # Load an image")
        print("  img = cv2.imread('your_image.jpg')")
        print()
        print("  # Run recognition")
        print("  results = recognizer([img])")
        print()
        print("  # Print results")
        print("  for text, confidence in results:")
        print("      print(f'Text: {text}, Confidence: {confidence:.4f}')")
        print()

    except Exception as e:
        print(f"⚠ Could not initialize recognizer: {e}")
        print("  Make sure PaddlePaddle is installed:")
        print("  pip install paddlepaddle")
else:
    print(f"⚠ Model directory not found: {model_dir}")
    print()
    print("To use the complete inference:")
    print("1. Download the EN PP-OCRv5 model:")
    print("   wget https://paddleocr.bj.bcebos.com/PP-OCRv5/english/en_PP-OCRv5_mobile_rec_infer.tar")
    print("   tar -xf en_PP-OCRv5_mobile_rec_infer.tar")
    print()
    print("2. Run inference:")
    print("   python ocr_en_v5_inference_standalone.py \\")
    print("       --model_dir ./en_PP-OCRv5_mobile_rec_infer \\")
    print("       --image_path your_image.jpg")

print()
print()

# ============================================================================
# Demo 4: Post-processing with Word Boxes
# ============================================================================
print("[Demo 4] Post-processing with Word-Level Bounding Boxes")
print("-" * 80)

# Create predictions for text with multiple words
target_text = "HELLO WORLD"
print(f"Target text: '{target_text}'")
print()

# Create prediction sequence
time_steps = len(target_text) * 2 + 5
preds = np.zeros((1, time_steps, num_classes))
sequence = [blank_idx]

for char in target_text:
    if char == ' ':
        char_idx = char_to_idx.get(' ', blank_idx)
    else:
        char_idx = char_to_idx.get(char, blank_idx)
    sequence.extend([char_idx, blank_idx])

while len(sequence) < time_steps:
    sequence.append(blank_idx)
sequence = sequence[:time_steps]

for t, char_idx in enumerate(sequence):
    preds[0, t, char_idx] = 0.88
    preds[0, t, :] += np.random.rand(num_classes) * 0.05

preds = preds / preds.sum(axis=2, keepdims=True)

# Decode with word box information
results = postprocessor(preds, return_word_box=True, wh_ratio_list=[1.0], max_wh_ratio=1.0)

if results:
    text, conf, word_info = results[0]
    print(f"Decoded text: '{text}'")
    print(f"Confidence: {conf:.4f}")
    print()

    if len(word_info) > 1:
        _, word_list, word_col_list, state_list = word_info
        print(f"Word-level information:")
        for i, (word, positions, state) in enumerate(zip(word_list, word_col_list, state_list)):
            word_text = ''.join(word)
            print(f"  Word {i+1}: '{word_text}' (type: {state})")
            print(f"    Positions: {positions}")

print()
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("Demo Summary")
print("=" * 80)
print()
print("✓ Post-processing module works independently")
print("✓ CTC label decoding with confidence scores")
print("✓ Support for word-level bounding boxes")
print("✓ Ready for integration with trained models")
print()
print("For complete documentation, see: OCR_EN_V5_README.md")
print()
print("Key Features:")
print("  • Standalone inference without full PaddleOCR installation")
print("  • Extracted post-processing methods for custom pipelines")
print("  • Support for PP-OCRv5 EN character recognition")
print("  • Easy integration with existing systems")
print()
print("=" * 80)
