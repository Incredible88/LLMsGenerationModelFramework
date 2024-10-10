# LLMsGenerationModelFramework
## Decoder-only model

![image](https://github.com/user-attachments/assets/358e6825-fbb7-4c0f-94f8-f31c7c5d53ea)


sequence: 50000  
classes: 100  
max length：512  
embedding dim: 512  
decoder layer: 6  

### Result:
![image](https://github.com/user-attachments/assets/1c9685b8-fe39-417b-89b4-a8140c8c5a45)
在50000样本，100类的数据测试中，即使loss一直在降低，  
准确率一直在一个很低的水平，没有过 5%，我怀疑是否是模型是否合适

# LSTM-Attention model

![image](https://github.com/user-attachments/assets/7c3ea8c5-32a0-4a96-8e39-ecdeb0581a4a)

在LSTM层 之后应用Attention，在得到LSTM的输出后使用批量矩阵乘法（torch.bmm）计算每个时间步的隐藏状态与所有其他时间步的隐藏状态之间的相似度得分。 
再进行归一化之后在于 LSTM 原始输出相乘得到加权的隐藏状态，最后输出的是最后一个时间步的输出 attn_out[:, -1, :]  

### Result:
![image](https://github.com/user-attachments/assets/e21a9be5-386a-4601-8e9a-668cbf6d1d33)
![image](https://github.com/user-attachments/assets/6da601bc-939e-4a32-9243-544fca938e9b)

### How to run
```bash
python decoder-only.py
python LSTM.py
