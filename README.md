# 环境配置指南

## 系统要求
- Python 3.8
- Git LFS 已安装（用于大文件管理）
- Conda 包管理器  
- 推荐使用 WSL2 和 Ubuntu

## 配置步骤

### 1. 创建 Conda 环境
```bash
# 创建名为 whisper 的 Python 3.8 环境
conda create -n whisper python=3.8 -y

# 激活环境
conda activate whisper
```

### 2. 安装基础依赖
```bash
# 安装 PyTorch 和相关库（自动选择兼容版本）
conda install pytorch=1.5 torchaudio cudatoolkit=10.2 -c pytorch

# 安装其他必要库
pip install datasets>=1.18.0 librosa jiwer evaluate
```

### 3. 安装特定版本 Transformers
```bash
# 安装指定版本的 Transformers 库
pip install transformers==4.32.0

# 修复 requests 安全依赖
pip install -U requests[security]
```

### 4. 下载大模型及LFS管理
```bash
# 克隆 Whisper 大模型仓库
git clone https://huggingface.co/openai/whisper-large-v2

# 进入模型目录
cd whisper-large-v2

# 初始化 Git LFS（必须在仓库目录内执行）
git lfs install

# 获取大文件（需确保网络稳定）
git lfs fetch

# 检出大文件内容
git lfs checkout  
```

### 5. 验证安装
```bash
# 检查关键库版本
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 检查模型文件（需在模型目录执行）
ls whisper-large-v2
```

### 6.下载数据集  
```bash
# 官网下载
wget https://mirrors.tuna.tsinghua.edu.cn/openslr/resources/33/data_aishell.tgz
tar -xzvf data_aishell.tgz

# 提取音频文件
./extract_all_wavs_absolute.sh
```

## 注意事项
1. **顺序敏感**：必须严格按步骤4的顺序执行，先克隆再操作LFS
2. **路径要求**：`git lfs install` 必须在克隆后的仓库目录内执行
3. **网络要求**：下载大文件时建议使用稳定网络，若中断可重新运行 `git lfs fetch`
4. **存储空间**：确保至少有 3GB 可用磁盘空间
5. **权限问题**：Linux/macOS 可能需要前缀 `sudo` 执行 LFS 命令  

# 训练模型  
```
python3 train.py
```

# 测试模型
```
python3 predict.py
```