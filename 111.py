import datasets
from datasets import *

# todo:cache_dir =指定下载位置
# datasets.load_dataset("madao33/new-title-chinese", cache_dir = r"E:\PROJECT\25_920_\00data\new-title-chinese" )
datasets = datasets.load_dataset(path=r"E:\PROJECT\25_920_\00data\new-title-chinese")
# todo:访问方式
# print(datasets["train"][0])
# todo:划分方式
# dataset["train"].select(range(5)) # 取前5条

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
    print("model_inputs")
    print(model_inputs)
    labels = tokenizer(example["title"], max_length=32, truncation=True)
    print("labels")
    print(labels)
    # label就是title编码的结果
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


datasets["train"] = datasets["train"].select(range(5))
datasets["validation"] = datasets["validation"].select(range(5))
for i in datasets["train"]:
    print(i)
print(datasets)
processed_datasets = datasets.map(preprocess_function)
print(processed_datasets)