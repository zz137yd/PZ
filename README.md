# PZ
This is a recorded Korean voice that can convert speech to text and identify the speaker's age, gender, and accent.

However, due to the poor quality of the dataset, the current model is not very accurate.

---

## 训练过程

1. **数据准备**  
   - 收集并标注韩语语音数据，包括年龄、性别和口音信息。
   - 对音频进行预处理（如降噪、分割等）。

2. **特征提取**  
   - 提取MFCC等音频特征。

3. **模型训练**  
   - 使用深度学习模型（如CNN、RNN等）进行训练。
   - 训练过程中采用验证集评估和参数优化。

4. **模型评估**  
   - 在测试集上评估准确率，调整模型参数。

---

## 一键启动网页界面

训练完成后，直接运行以下命令即可自动创建网页，检查和演示整个项目功能：
