# 快速开始指南 (Quick Start Guide)

## OCR EN V5 字符识别 - 独立推理脚本

本指南帮助您快速开始使用PP-OCRv5英文字符识别的独立推理脚本。

---

## 🚀 三步快速开始

### 第一步：安装依赖

```bash
pip install paddlepaddle opencv-python numpy pillow
```

### 第二步：下载模型

```bash
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/english/en_PP-OCRv5_mobile_rec_infer.tar
tar -xf en_PP-OCRv5_mobile_rec_infer.tar
```

### 第三步：运行推理

```bash
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path your_image.jpg
```

---

## 📝 使用示例

### 示例1：命令行使用

```bash
# 识别单张图片
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path demo.jpg

# 批量识别
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path ./images/

# 使用GPU
python ocr_en_v5_inference_standalone.py \
    --model_dir ./en_PP-OCRv5_mobile_rec_infer \
    --image_path demo.jpg \
    --use_gpu
```

### 示例2：Python代码

```python
from ocr_en_v5_inference_standalone import OCRENV5Recognizer
import cv2

# 初始化
recognizer = OCRENV5Recognizer(
    model_dir='./en_PP-OCRv5_mobile_rec_infer'
)

# 识别
img = cv2.imread('demo.jpg')
results = recognizer([img])

# 输出
for text, conf in results:
    print(f'{text} (置信度: {conf:.4f})')
```

### 示例3：仅使用后处理

```python
from rec_postprocess_standalone import CTCLabelDecode
import numpy as np

# 初始化后处理器
postprocessor = CTCLabelDecode(
    character_dict_path='./ppocr/utils/dict/ppocrv5_en_dict.txt',
    use_space_char=True
)

# 处理模型输出 (shape: [batch, time_steps, num_classes])
results = postprocessor(model_predictions)

# 获取结果
for text, confidence in results:
    print(f'{text}: {confidence:.4f}')
```

---

## 🧪 运行演示

我们提供了多个演示脚本：

### 1. 后处理演示
```bash
python rec_postprocess_standalone.py
```
展示独立的后处理功能，包括CTC解码和置信度计算。

### 2. 完整演示
```bash
python demo_ocr_en_v5.py
```
展示多个使用场景：
- 后处理独立使用
- 真实CTC预测
- 完整推理流程
- 单词级边界框

### 3. 简单示例
```bash
python example_simple.py
```
最简单的使用示例和代码模板。

---

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `ocr_en_v5_inference_standalone.py` | 完整推理脚本（含模型加载、预处理、推理、后处理） |
| `rec_postprocess_standalone.py` | 独立后处理脚本（CTC解码） |
| `demo_ocr_en_v5.py` | 完整演示脚本 |
| `example_simple.py` | 简单使用示例 |
| `OCR_EN_V5_README.md` | 详细文档（中英文） |
| `QUICKSTART.md` | 本快速开始指南 |

---

## 🔧 配置参数

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | 必需 | 模型目录 |
| `--image_path` | 必需 | 图片路径 |
| `--dict_path` | `./ppocr/utils/dict/ppocrv5_en_dict.txt` | 字典路径 |
| `--use_gpu` | `False` | 使用GPU |
| `--save_results` | `None` | 保存结果文件 |

---

## ❓ 常见问题

### Q1: 缺少PaddlePaddle？
```bash
# CPU版本
pip install paddlepaddle

# GPU版本 (CUDA 11.2)
pip install paddlepaddle-gpu
```

### Q2: 缺少OpenCV？
```bash
pip install opencv-python
```

### Q3: 模型在哪里下载？
访问: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md

### Q4: 能识别其他语言吗？
这个脚本专门针对英文V5模型。对于其他语言，需要：
1. 下载对应语言的模型
2. 使用对应的字典文件
3. 参数基本相同

### Q5: 如何提高识别准确率？
- 使用高分辨率图片
- 确保文字区域清晰
- 使用文本检测模型先裁剪文字区域
- 考虑使用server版本模型（更大、更准）

---

## 📊 性能参考

**Mobile模型 (en_PP-OCRv5_mobile_rec)**
- 大小: ~10MB
- 速度: 快
- 适用: 移动设备、实时应用

**Server模型 (en_PP-OCRv5_server_rec)**
- 大小: ~80MB
- 速度: 中等
- 精度: 更高
- 适用: 服务器端、高精度要求

---

## 🎯 下一步

1. ✅ 阅读完整文档: `OCR_EN_V5_README.md`
2. ✅ 运行演示脚本: `python demo_ocr_en_v5.py`
3. ✅ 查看简单示例: `python example_simple.py`
4. ✅ 开始您的项目！

---

## 📞 支持

- 问题反馈: https://github.com/PaddlePaddle/PaddleOCR/issues
- 完整文档: https://github.com/PaddlePaddle/PaddleOCR
- 模型列表: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md

---

## 📄 许可证

Apache License 2.0

---

**祝使用愉快！ Happy Coding! 🎉**
