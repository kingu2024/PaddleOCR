#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Post-Processing Script for OCR Recognition
Extracted from PaddleOCR ppocr/postprocess/rec_postprocess.py

This script provides post-processing methods for OCR character recognition,
specifically for EN PP-OCRv5 models using CTCLabelDecode.

Usage:
    from rec_postprocess_standalone import CTCLabelDecode

    # Initialize post-processor
    postprocessor = CTCLabelDecode(
        character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
        use_space_char=True
    )

    # Process model predictions
    results = postprocessor(preds)
"""

import os
import numpy as np
import re


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        """Reverse prediction for Arabic text"""
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        """Add special characters to dictionary (to be overridden by subclasses)"""
        return dict_character

    def get_word_info(self, text, selection):
        """
        Group the decoded characters and record the corresponding decoded positions.

        Args:
            text: the decoded text
            selection: the bool array that identifies which columns of features are decoded as non-separated characters
        Returns:
            word_list: list of the grouped words
            word_col_list: list of decoding positions corresponding to each character in the grouped word
            state_list: list of marker to identify the type of grouping words, including two types:
                        - 'cn': continuous chinese characters (e.g., 你好啊)
                        - 'en&num': continuous english characters (e.g., hello), number (e.g., 123, 1.123),
                                    or mixed of them connected by '-' (e.g., VGG-16)
                        The remaining characters in text are treated as separators between groups.
        """
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            # Use \w with UNICODE flag to match letters (including accented chars) and digits
            # Exclude underscore since \w includes it but we want to treat it as splitter
            elif bool(re.search(r"[\w]", char, re.UNICODE)) and char != "_":
                c_state = "en&num"
            else:
                c_state = "splitter"

            # Handle apostrophes in French words like "n'êtes"
            if char == "'" and state == "en&num":
                c_state = "en&num"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ):  # grouping floating number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ):  # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"

            if state == None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """
        Convert text-index into text-label.

        Args:
            text_index: numpy array of predicted character indices
            text_prob: numpy array of prediction probabilities
            is_remove_duplicate: whether to remove duplicate characters
            return_word_box: whether to return word bounding box information

        Returns:
            result_list: list of (text, confidence) tuples or (text, confidence, word_info) tuples
        """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        """Get list of token indices to ignore (blank token for CTC)"""
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """
    CTC Label Decoder for converting CTC predictions to text.

    This is the primary post-processing method for PP-OCRv5 EN recognition models.
    It converts raw CTC output probabilities to readable text with confidence scores.

    Args:
        character_dict_path: path to character dictionary file
        use_space_char: whether to use space character in recognition

    Example:
        >>> postprocessor = CTCLabelDecode(
        ...     character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
        ...     use_space_char=True
        ... )
        >>> # preds shape: [batch_size, time_steps, num_classes]
        >>> results = postprocessor(preds)
        >>> for text, conf in results:
        ...     print(f"Text: {text}, Confidence: {conf:.4f}")
    """

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, return_word_box=False, *args, **kwargs):
        """
        Post-process CTC predictions.

        Args:
            preds: Model predictions, numpy array of shape [batch_size, time_steps, num_classes]
                   or list/tuple of predictions (will use the last one)
            label: Optional ground truth labels for evaluation
            return_word_box: Whether to return word-level bounding box information

        Returns:
            list of tuples: [(text, confidence), ...] or [(text, confidence, word_info), ...]
        """
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]

        # Convert to numpy if needed
        if hasattr(preds, 'numpy'):  # Check if it's a tensor with numpy() method
            preds = preds.numpy()

        # Get predicted indices and probabilities
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        # Decode to text
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )

        # Adjust word positions if word boxes are requested
        if return_word_box:
            for rec_idx, rec in enumerate(text):
                wh_ratio = kwargs.get("wh_ratio_list", [1.0] * len(text))[rec_idx]
                max_wh_ratio = kwargs.get("max_wh_ratio", 1.0)
                rec[2][0] = rec[2][0] * (wh_ratio / max_wh_ratio)

        if label is None:
            return text

        # If labels provided, decode them too for evaluation
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        """Add blank token for CTC at the beginning of character list"""
        dict_character = ["blank"] + dict_character
        return dict_character


# Example usage
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("OCR Recognition Post-Processing - Standalone Script")
    print("=" * 80)

    # Example 1: Initialize post-processor with default dictionary
    print("\n[Example 1] Initialize CTCLabelDecode with default dictionary")
    postprocessor = CTCLabelDecode(use_space_char=False)
    print(f"Number of characters: {len(postprocessor.character)}")
    print(f"First 10 characters: {postprocessor.character[:10]}")

    # Example 2: Initialize with English V5 dictionary
    dict_path = "./ppocr/utils/dict/ppocrv5_en_dict.txt"
    if os.path.exists(dict_path):
        print(f"\n[Example 2] Initialize CTCLabelDecode with EN V5 dictionary")
        postprocessor = CTCLabelDecode(
            character_dict_path=dict_path,
            use_space_char=True
        )
        print(f"Number of characters: {len(postprocessor.character)}")
        print(f"Character dict loaded from: {dict_path}")
    else:
        print(f"\n[Example 2] EN V5 dictionary not found at {dict_path}")
        print("Using default dictionary instead")

    # Example 3: Simulate CTC predictions and decode
    print("\n[Example 3] Simulate CTC predictions and decode")
    batch_size = 2
    time_steps = 20
    num_classes = len(postprocessor.character)

    # Create random predictions (simulating model output)
    np.random.seed(42)
    fake_preds = np.random.rand(batch_size, time_steps, num_classes)

    # Process predictions
    results = postprocessor(fake_preds)

    print(f"Batch size: {batch_size}")
    print(f"Results:")
    for idx, (text, conf) in enumerate(results):
        print(f"  Sample {idx+1}: '{text}' (confidence: {conf:.4f})")

    # Example 4: Demonstrate with more realistic predictions
    print("\n[Example 4] Create more realistic predictions for 'HELLO'")
    # Map characters to indices
    char_to_idx = postprocessor.dict
    blank_idx = 0

    # Create prediction sequence for "HELLO"
    target_text = "HELLO"
    time_steps = 15
    preds = np.zeros((1, time_steps, num_classes))

    # Set high probabilities for target characters
    sequence = [blank_idx, char_to_idx.get('H', 1), blank_idx, char_to_idx.get('E', 1),
                blank_idx, char_to_idx.get('L', 1), char_to_idx.get('L', 1), blank_idx,
                char_to_idx.get('O', 1), blank_idx] + [blank_idx] * 5

    for t, char_idx in enumerate(sequence[:time_steps]):
        preds[0, t, char_idx] = 0.95
        # Add small noise to other classes
        preds[0, t, :] += np.random.rand(num_classes) * 0.05

    # Normalize to make it look like probabilities
    preds = preds / preds.sum(axis=2, keepdims=True)

    results = postprocessor(preds)
    decoded_text, conf = results[0]
    print(f"Target text: '{target_text}'")
    print(f"Decoded text: '{decoded_text}' (confidence: {conf:.4f})")

    print("\n" + "=" * 80)
    print("Post-processing script can be imported and used independently!")
    print("=" * 80)
