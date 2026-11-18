# Classification Workbench

用表格、图像和中文短文本三种数据练习分类建模。每个实验都包含数据准备、验证集选参和测试集评估，可从命令行运行并导出指标和图表。

| 任务 | 比较内容 | 从这里读起 |
| --- | --- | --- |
| 银行贷款审批 | scikit-learn 决策树与手写 ID3，观察收入离散化和预剪枝的影响 | [银行实验](docs/methods.md#银行贷款审批) |
| 手写数字识别 | PyTorch MLP 的宽度、深度、激活函数、批量大小和 Dropout，以及联合调参、数据增强和 CNN | [数字实验](docs/methods.md#手写数字识别) |
| 中文新闻标题分类 | 手写与库版多项式朴素贝叶斯，比较词袋、TF-IDF 和线性 SVM | [新闻实验](docs/methods.md#中文新闻标题分类) |

## 快速开始

推荐使用 Python 3.11 至 3.14。在项目根目录创建并激活虚拟环境。

| 平台 | 创建 | 激活 |
| --- | --- | --- |
| Windows PowerShell | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

安装通用依赖并运行银行、新闻两个离线演示。

~~~bash
python -m pip install -r requirements.txt
python -m classification_workbench bank --demo
python -m classification_workbench news --demo
~~~

手写数字实验还需要安装 PyTorch 与 torchvision。Windows 或 Linux 的 CPU 环境可以使用以下命令。macOS 和 CUDA 安装方式见 [PyTorch 安装指南](https://docs.pytorch.org/get-started/)。

~~~bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m classification_workbench digits --demo --device cpu
~~~

银行和新闻演示使用程序生成的数据。数字演示使用随 scikit-learn 安装的 8×8 手写数字图像，并将其缩放到 28×28。安装依赖后，这三条演示命令均可离线运行。完整基准的结果见 [结果与分析](docs/results.md)。

## 使用基准数据

~~~bash
python -m classification_workbench bank --data-dir data/bank
python -m classification_workbench digits --sweep --extensions
python -m classification_workbench news --data-dir data/news
~~~

- 银行和新闻数据从本地读取，文件名和列格式见 [数据准备](docs/methods.md)。MNIST 由 torchvision 下载到 `data/digits/`，后续运行复用本地缓存。
- 数字命令默认只训练 MLP 基线。`--sweep` 比较单变量配置，`--extensions` 增加四组扩展配置。可以用 `--epochs 1` 快速检查流程，正式对照使用默认的 5 轮训练。
- 每次运行生成 `metrics.json` 和图表，默认写到 `outputs/` 下。可通过 `--out` 指定目录，通过 `--seed` 更换随机种子。

## 源码入口

- [id3.py](classification_workbench/id3.py) 计算信息增益并递归构建多分支树。
- [digits.py](classification_workbench/digits.py) 定义网络、训练循环和对照配置。
- [naive_bayes.py](classification_workbench/naive_bayes.py) 直接在稀疏词频矩阵上估计多项式朴素贝叶斯。

运行 `python -m unittest discover -s tests -v` 可检查手写模型和评价指标。
