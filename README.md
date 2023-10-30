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

## 数据


全量推文数据：https://www.kaggle.com/datasets/jachinlin2022/twitter-and-reddit-post-from-2020-01-012023-05-31
- pred, scores 情感分析结果
- label 过滤模型结果，0为价格不相关，1为价格相关
- full_text 推文正文

## 流程
1. 使用bitcoin_scrapy爬取Twitter数据，在main.ipynb对数据remove_link并对正文去重
2. 训练过滤模型，使用FilterModel\data\filter_model_data.csv作为训练集，FilterModel\data\filter_model_manual_400.csv作为测试集。
3. 使用过滤模型对（1）爬取的数据进行过滤
4. 训练情感模型，使用EmotionModel\train_data_3_class_clean.jsonl指令微调
5. 使用情感模型对(3)过滤后的数据进行情感分析
6. 在EmotionModel\get_best_corr.ipynb中，对（5）分析后的数据进行点赞转发过滤后daily聚合为index
7. 在EmotionModel\regression.ipynb中，对(6)聚合后的数据进行自回归、线性回归、因果检验


