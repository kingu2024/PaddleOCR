#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for model loading logic
Tests the enhanced model file detection and error handling
"""

import os
import sys

def test_find_model_files(model_dir):
    """
    Test the model file finding logic (extracted from _create_predictor)
    """
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
        print(f"✓ Found model structure: {file_name}.json (new format)")
    elif os.path.exists(pdmodel_path):
        model_file_path = pdmodel_path
        print(f"✓ Found model structure: {file_name}.pdmodel")
    else:
        available_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
        raise FileNotFoundError(
            f"Model structure file not found in: {model_dir}\n"
            f"Expected: '{file_name}.pdmodel' or '{file_name}.json'\n"
            f"Found parameters: {params_file_path}\n"
            f"Available files: {available_files}\n\n"
            f"Note: A complete model requires both structure (.pdmodel or .json) and parameters (.pdiparams) files.\n"
            f"Please ensure you have downloaded the complete model package."
        )

    return model_file_path, params_file_path


def run_tests():
    """Run all test cases"""
    print("=" * 80)
    print("Testing Model File Detection Logic")
    print("=" * 80)
    print()

    # Test 1: Non-existent directory
    print("[Test 1] Non-existent directory")
    print("-" * 80)
    try:
        test_find_model_files("/tmp/nonexistent_dir_12345")
        print("✗ FAILED: Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print("✓ PASSED: Properly detected missing directory")
        print(f"Error message: {str(e)[:150]}...")
    print()

    # Test 2: Directory with only .pdiparams (incomplete model)
    print("[Test 2] Incomplete model (only .pdiparams)")
    print("-" * 80)
    test_dir = "/tmp/test_incomplete_model"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/inference.pdiparams", "w") as f:
        f.write("dummy")

    try:
        test_find_model_files(test_dir)
        print("✗ FAILED: Should have raised FileNotFoundError for missing .pdmodel/.json")
    except FileNotFoundError as e:
        error_msg = str(e)
        print("✓ PASSED: Detected incomplete model")
        print(f"  - Mentions 'complete model': {'complete model' in error_msg}")
        print(f"  - Shows available files: {'Available files' in error_msg}")
        print(f"  - Gives helpful guidance: {'Please ensure' in error_msg}")
    print()

    # Test 3: Complete model with .pdmodel
    print("[Test 3] Complete model with .pdmodel")
    print("-" * 80)
    test_dir = "/tmp/test_complete_pdmodel"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/inference.pdiparams", "w") as f:
        f.write("dummy")
    with open(f"{test_dir}/inference.pdmodel", "w") as f:
        f.write("dummy")

    try:
        model_file, params_file = test_find_model_files(test_dir)
        print("✓ PASSED: Successfully found model files")
        print(f"  Model structure: {os.path.basename(model_file)}")
        print(f"  Model parameters: {os.path.basename(params_file)}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    print()

    # Test 4: Complete model with .json (new format)
    print("[Test 4] Complete model with .json (new format)")
    print("-" * 80)
    test_dir = "/tmp/test_complete_json"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/inference.pdiparams", "w") as f:
        f.write("dummy")
    with open(f"{test_dir}/inference.json", "w") as f:
        f.write("{}")

    try:
        model_file, params_file = test_find_model_files(test_dir)
        print("✓ PASSED: Successfully found model files")
        print(f"  Model structure: {os.path.basename(model_file)}")
        print(f"  Model parameters: {os.path.basename(params_file)}")
        assert ".json" in model_file, "Should prefer .json over .pdmodel"
        print("  ✓ Correctly prefers .json format")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    print()

    # Test 5: Alternative naming (model.* instead of inference.*)
    print("[Test 5] Alternative naming (model.* prefix)")
    print("-" * 80)
    test_dir = "/tmp/test_model_prefix"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/model.pdiparams", "w") as f:
        f.write("dummy")
    with open(f"{test_dir}/model.pdmodel", "w") as f:
        f.write("dummy")

    try:
        model_file, params_file = test_find_model_files(test_dir)
        print("✓ PASSED: Successfully found model files with 'model' prefix")
        print(f"  Model structure: {os.path.basename(model_file)}")
        print(f"  Model parameters: {os.path.basename(params_file)}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    print()

    # Test 6: Both .json and .pdmodel exist
    print("[Test 6] Both .json and .pdmodel exist (should prefer .json)")
    print("-" * 80)
    test_dir = "/tmp/test_both_formats"
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/inference.pdiparams", "w") as f:
        f.write("dummy")
    with open(f"{test_dir}/inference.pdmodel", "w") as f:
        f.write("dummy")
    with open(f"{test_dir}/inference.json", "w") as f:
        f.write("{}")

    try:
        model_file, params_file = test_find_model_files(test_dir)
        print("✓ PASSED: Successfully found model files")
        print(f"  Model structure: {os.path.basename(model_file)}")
        if ".json" in model_file:
            print("  ✓ Correctly prefers .json over .pdmodel")
        else:
            print("  ✗ Should prefer .json but selected .pdmodel")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    print()

    print("=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
