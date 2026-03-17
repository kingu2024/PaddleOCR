# OCR EN V5 字符识别推理代码 (Character Recognition Inference Code)

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文文档

本项目提供了PP-OCRv5英文字符识别模型的独立推理代码和后处理方法。代码已从PaddleOCR源码中提取并可独立运行。

### 📁 文件说明

1. **`ocr_en_v5_inference_standalone.py`** - 完整的推理脚本
   - 模型加载
   - 图像预处理
   - 推理执行
   - 结果后处理

2. **`rec_postprocess_standalone.py`** - 独立的后处理脚本
   - CTCLabelDecode 类
   - BaseRecLabelDecode 基类
   - 字符解码和置信度计算

### 🚀 快速开始

#### 安装依赖

```bash
pip install paddlepaddle opencv-python numpy pillow
# 或使用GPU版本
pip install paddlepaddle-gpu opencv-python numpy pillow
```

#### 下载模型

从PaddleOCR模型库下载EN PP-OCRv5模型：
- 模型列表: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md
- 英文识别模型: en_PP-OCRv5_mobile_rec

```bash
# 下载英文识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/english/en_PP-OCRv5_mobile_rec_infer.tar
tar -xf en_PP-OCRv5_mobile_rec_infer.tar
```

### 💡 使用方法

#### 方法1: 命令行使用

```bash
# 基本用法 - 识别单张图片
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./demo.jpg

# 批量处理
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./images/

# 使用GPU加速
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./demo.jpg \
    --use_gpu

# 保存结果到文件
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./images/ \
    --save_results ./results.txt
```

#### 方法2: Python代码调用

```python
from ocr_en_v5_inference_standalone import OCRENV5Recognizer
import cv2

# 初始化识别器
recognizer = OCRENV5Recognizer(
    model_dir='./en_PP-OCRv5_mobile_rec_infer',
    dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
    use_gpu=False
)

# 加载图片
img = cv2.imread('demo.jpg')

# 执行识别
results = recognizer([img])

# 输出结果
for text, confidence in results:
    print(f'识别文本: {text}')
    print(f'置信度: {confidence:.4f}')
```

#### 方法3: 仅使用后处理模块

```python
from rec_postprocess_standalone import CTCLabelDecode
import numpy as np

# 初始化后处理器
postprocessor = CTCLabelDecode(
    character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
    use_space_char=True
)

# 假设你已经有了模型的输出 (shape: [batch_size, time_steps, num_classes])
model_output = np.random.rand(1, 20, 439)  # 示例数据

# 执行后处理
results = postprocessor(model_output)

for text, confidence in results:
    print(f'文本: {text}, 置信度: {confidence:.4f}')
```

### 📊 参数说明

#### 推理脚本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | str | 必需 | 推理模型目录路径 |
| `--dict_path` | str | `./ppocr/utils/dict/ppocrv5_en_dict.txt` | 字符字典路径 |
| `--image_path` | str | None | 输入图片路径或目录 |
| `--use_gpu` | bool | False | 是否使用GPU |
| `--use_space_char` | bool | True | 是否识别空格字符 |
| `--rec_image_shape` | str | `3,48,320` | 识别图像形状 (C,H,W) |
| `--save_results` | str | None | 保存结果的文件路径 |

#### 后处理类参数

**CTCLabelDecode**:
- `character_dict_path`: 字符字典文件路径
- `use_space_char`: 是否使用空格字符
- `return_word_box`: 是否返回单词级边界框信息

### 🔧 技术细节

#### 模型架构
- **算法**: SVTR_LCNet
- **骨干网络**: PPLCNetV3 (scale=0.95)
- **头部**: MultiHead (CTCHead + NRTRHead，推理时仅使用CTC)
- **输入形状**: [3, 48, 320] (C, H, W)
- **字符集**: ppocrv5_en_dict.txt (437个字符)

#### 预处理流程
1. 按宽高比调整图片尺寸
2. 保持宽高比resize到目标高度
3. 归一化到 [-1, 1]
4. 用零填充到目标宽度

#### 后处理流程
1. 取最大概率的字符索引 (argmax)
2. 移除重复字符 (CTC去重)
3. 移除blank标记
4. 转换为文本
5. 计算置信度

### 📝 输出格式

命令行输出示例:
```
[1/3] Processing: ./demo.jpg
  Image shape: (48, 320, 3)
  Result: 'HELLO WORLD'
  Confidence: 0.9823
```

保存的结果文件格式 (TSV):
```
./demo.jpg    HELLO WORLD    0.9823
./test.jpg    PADDLEOCR      0.9654
```

### 🔍 测试后处理脚本

```bash
# 运行后处理脚本的示例代码
python rec_postprocess_standalone.py
```

这将运行内置的示例，展示:
1. 初始化CTCLabelDecode
2. 加载字符字典
3. 模拟CTC预测并解码
4. 输出识别结果和置信度

### 🎯 应用场景

- 文档OCR
- 场景文字识别
- 票据识别
- 车牌识别
- 自定义OCR应用

### ⚠️ 注意事项

1. 确保模型文件存在: `inference.pdmodel` 和 `inference.pdiparams`
2. 字典文件必须与模型训练时使用的字典一致
3. 输入图片应为包含单行或少量文字的裁剪图片
4. 对于复杂文档，建议先使用文本检测模型

### 📚 参考资料

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 技术报告](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/PP-OCRv5_introduction.md)
- [模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)

---

<a name="english"></a>
## English Documentation

This project provides standalone inference code and post-processing methods for PP-OCRv5 English character recognition model. The code has been extracted from PaddleOCR source code and can run independently.

### 📁 File Description

1. **`ocr_en_v5_inference_standalone.py`** - Complete inference script
   - Model loading
   - Image preprocessing
   - Inference execution
   - Result post-processing

2. **`rec_postprocess_standalone.py`** - Standalone post-processing script
   - CTCLabelDecode class
   - BaseRecLabelDecode base class
   - Character decoding and confidence calculation

### 🚀 Quick Start

#### Install Dependencies

```bash
pip install paddlepaddle opencv-python numpy pillow
# Or use GPU version
pip install paddlepaddle-gpu opencv-python numpy pillow
```

#### Download Model

Download EN PP-OCRv5 model from PaddleOCR model zoo:
- Model List: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
- English Recognition Model: en_PP-OCRv5_mobile_rec

```bash
# Download English recognition model
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/english/en_PP-OCRv5_mobile_rec_infer.tar
tar -xf en_PP-OCRv5_mobile_rec_infer.tar
```

### 💡 Usage

#### Method 1: Command Line

```bash
# Basic usage - recognize single image
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./demo.jpg

# Batch processing
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./images/

# Use GPU acceleration
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./demo.jpg \
    --use_gpu

# Save results to file
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./images/ \
    --save_results ./results.txt
```

#### Method 2: Python Code

```python
from ocr_en_v5_inference_standalone import OCRENV5Recognizer
import cv2

# Initialize recognizer
recognizer = OCRENV5Recognizer(
    model_dir='./en_PP-OCRv5_mobile_rec_infer',
    dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
    use_gpu=False
)

# Load image
img = cv2.imread('demo.jpg')

# Run recognition
results = recognizer([img])

# Print results
for text, confidence in results:
    print(f'Recognized Text: {text}')
    print(f'Confidence: {confidence:.4f}')
```

#### Method 3: Post-processing Only

```python
from rec_postprocess_standalone import CTCLabelDecode
import numpy as np

# Initialize post-processor
postprocessor = CTCLabelDecode(
    character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
    use_space_char=True
)

# Assume you already have model output (shape: [batch_size, time_steps, num_classes])
model_output = np.random.rand(1, 20, 439)  # Example data

# Run post-processing
results = postprocessor(model_output)

for text, confidence in results:
    print(f'Text: {text}, Confidence: {confidence:.4f}')
```

### 📊 Parameters

#### Inference Script Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_dir` | str | Required | Inference model directory path |
| `--dict_path` | str | `./ppocr/utils/dict/ppocrv5_en_dict.txt` | Character dictionary path |
| `--image_path` | str | None | Input image path or directory |
| `--use_gpu` | bool | False | Whether to use GPU |
| `--use_space_char` | bool | True | Whether to recognize space character |
| `--rec_image_shape` | str | `3,48,320` | Recognition image shape (C,H,W) |
| `--save_results` | str | None | File path to save results |

#### Post-processing Class Parameters

**CTCLabelDecode**:
- `character_dict_path`: Character dictionary file path
- `use_space_char`: Whether to use space character
- `return_word_box`: Whether to return word-level bounding box info

### 🔧 Technical Details

#### Model Architecture
- **Algorithm**: SVTR_LCNet
- **Backbone**: PPLCNetV3 (scale=0.95)
- **Head**: MultiHead (CTCHead + NRTRHead, only CTC for inference)
- **Input Shape**: [3, 48, 320] (C, H, W)
- **Character Set**: ppocrv5_en_dict.txt (437 characters)

#### Preprocessing Pipeline
1. Calculate aspect ratio
2. Resize maintaining aspect ratio to target height
3. Normalize to [-1, 1]
4. Pad with zeros to target width

#### Post-processing Pipeline
1. Get character indices with max probability (argmax)
2. Remove duplicate characters (CTC deduplication)
3. Remove blank tokens
4. Convert to text
5. Calculate confidence

### 📝 Output Format

Command line output example:
```
[1/3] Processing: ./demo.jpg
  Image shape: (48, 320, 3)
  Result: 'HELLO WORLD'
  Confidence: 0.9823
```

Saved results file format (TSV):
```
./demo.jpg    HELLO WORLD    0.9823
./test.jpg    PADDLEOCR      0.9654
```

### 🔍 Test Post-processing Script

```bash
# Run the post-processing script's example code
python rec_postprocess_standalone.py
```

This will run built-in examples showing:
1. Initialize CTCLabelDecode
2. Load character dictionary
3. Simulate CTC predictions and decode
4. Output recognition results and confidence

### 🎯 Use Cases

- Document OCR
- Scene text recognition
- Receipt recognition
- License plate recognition
- Custom OCR applications

### ⚠️ Notes

1. Ensure model files exist: `inference.pdmodel` and `inference.pdiparams`
2. Dictionary file must match the one used during model training
3. Input images should be cropped images containing single line or small amount of text
4. For complex documents, text detection model is recommended first

### 📚 References

- [PaddleOCR Official Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 Technical Report](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/PP-OCRv5_introduction_en.md)
- [Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)

---

## 📄 License

This code is extracted from PaddleOCR and follows the Apache License 2.0.

## 🤝 Contributing

Issues and pull requests are welcome!

## 📧 Contact

For questions or issues, please visit: https://github.com/PaddlePaddle/PaddleOCR/issues
