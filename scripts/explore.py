import pandas as pd

train_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
train_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
train_s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print(train_s1.shape, train_s2.shape, train_s3.shape, ground_truth.shape)
print(train_s1.head())