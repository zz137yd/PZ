# PZ
This is a recorded Korean voice that can convert speech to text and identify the speaker's age, gender, and accent.

However, due to the poor quality of the dataset, the current model is not very accurate.

# PZ项目：韩语语音识别与说话人分析 (PZ Project: Korean Speech Recognition & Speaker Analysis)

本项目是一个基于深度学习的韩语语音处理系统，旨在实现两大核心功能：将韩语语音转换为文本（语音识别），并从语音中识别说话人的年龄、性别及口音（说话人分析）。

## 🌟 项目特点

*   **多任务学习**: 模型在一个统一的框架内同时处理语音内容转写和说话人特征分类。
*   **端到端架构**: 采用先进的端到端模型，直接从原始音频波形中学习特征，简化了传统语音处理的复杂流程。
*   **核心功能**:
    *   **语音转文本 (Speech-to-Text)**: 将输入的韩语语音流利地转换为文字稿。
    *   **说话人属性识别 (Speaker Attribute Recognition)**:
        *   年龄段 (Age)
        *   性别 (Gender)
        *   口音 (Accent)

## ⚠️ 当前模型性能与局限性

我们致力于项目的透明化。需要指出的是，当前版本的模型在准确率方面表现尚有不足。

*   **核心原因**: 主要受限于训练数据集的质量。我们观察到数据集中存在部分标注错误、背景噪音较多以及说话人特征分布不均等问题，这直接影响了模型的学习效果。
*   **具体表现**:
    *   **语音识别 (ASR)**: 词错误率 (Word Error Rate, WER) 相对较高，尤其是在处理带有浓重口音或在嘈杂环境下的语音时。
    *   **分类任务**: 年龄和口音的分类准确率有待提升，性别分类表现相对较好。

我们正在积极寻找更高质量的数据集，并计划在未来版本中对模型进行迭代优化。

## ⚙️ 训练流程详解

以下是本模型完整的训练步骤，为您详细揭示从数据到模型的全过程。

#### 1. 数据准备与预处理

*   **数据来源**: (*请在此处填写您的数据集来源，例如：AI-Hub韩语数据集、自录数据等*)。
*   **数据清洗**: 使用`ffmpeg`和`librosa`对原始音频进行标准化处理，包括：
    *   重采样至 16kHz。
    *   转换为单声道。
    *   去除音频文件前后的静音部分。
*   **特征提取**: 从音频中提取对数梅尔频谱图 (Log-Mel Spectrogram) 作为模型的输入特征。
*   **标签制作**:
    *   为每个音频文件准备对应的文本转写稿 (`transcript.txt`)。
    *   为每个音频文件准备包含年龄、性别、口音信息的元数据标签 (`metadata.csv`)。

#### 2. 模型架构

*   **骨干网络**: 模型主体采用 **Conformer** 架构 (*可替换为您使用的模型，如 Transformer, Wav2Vec 2.0等*)，它能高效地捕捉音频序列的局部和全局依赖关系。
*   **任务分支**:
    *   **ASR分支**: 在骨干网络的输出后连接一个 **CTC (Connectionist Temporal Classification)** 解码器，用于语音识别。
    *   **分类分支**: 将骨干网络的输出通过一个平均池化层（Mean Pooling），然后送入多个独立的全连接层（Dense Layers），分别用于年龄、性别和口音的分类。

#### 3. 训练环境

*   **框架**: PyTorch
*   **硬件**: (*请填写您使用的具体硬件，例如: Google Colab (NVIDIA T4 GPU), Lambda Labs (NVIDIA A100 GPU)等*)
*   **主要依赖库**: `torchaudio`, `librosa`, `transformers`, `pandas`, `numpy`

#### 4. 训练过程

*   **损失函数 (Loss Function)**:
    *   采用多任务损失，是 **CTC Loss** 和 **交叉熵损失 (Cross-Entropy Loss)** 的加权和。
    *   `Total Loss = α * CTC_Loss + β * CE_Loss_age + γ * CE_Loss_gender + δ * CE_Loss_accent`
*   **优化器 (Optimizer)**: 使用 **AdamW** 优化器，并配合学习率预热（Warmup）和衰减（Decay）策略。
*   **超参数 (Hyperparameters)**:
    *   **学习率 (Learning Rate)**: 1e-4
    *   **批次大小 (Batch Size)**: 16
    *   **训练周期 (Epochs)**: 50
    *   **混合精度训练**: 开启 (FP16)，以加速训练并减少显存占用。

#### 5. 评估

*   **评估指标**:
    *   **ASR**: 词错误率 (Word Error Rate, WER)
    *   **分类**: 准确率 (Accuracy) 和 F1分数 (F1-Score)
*   **评估集**: 从原始数据集中划分出10%作为独立的测试集，用于评估最终模型的泛化能力。

## 🔗 下载预训练模型

我们已经将当前训练好的模型权重上传，您可以通过以下链接下载。

*   **模型下载链接**: [**点击这里下载模型** (*在此处替换为您的Hugging Face、Google Drive或其他托管平台的链接*)]()
*   **文件内容**: 压缩包内包含模型权重文件 (`model.pt`) 和一个配置文件 (`config.json`)。

## 🚀 如何使用

下面是一个简单的示例，展示如何加载我们的模型并进行一次推理。
