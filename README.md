# 项目名称

该项目是bitcoin sentiment index代码目录，包含以下文件和文件夹：

- `.gitignore`：git忽略配置文件
- `bitcoin_scrapy`：比特币数据爬虫
- `EmotionModel`：情感模型
- `FilterModel`：过滤模型
- `PredictModel`：预测模型


## Filter Model
- data： 过滤模型训练集和测试集
- filter_model：基于hard prompt tuning的过滤模型
  - config.json：配置文件
  - train.py：训练模型
  - predict.py：使用训练好的模型对数据进行过滤
- SimCSE：基于对比学习的过滤模型
  - train.py：训练模型
  - run_unsup_example.sh：训练脚本
  - predict.py：使用训练好的模型对数据进行过滤
## Sentiment Model
- finetune.ipynb：使用LoRA指令微调llama模型
- analysis_daily.ipynb：按天进行相关性分析
- analysis_hourly.ipynb：按小时进行相关性分析
- data.ipynb:数据处理
- regression.ipynb：自回归分析