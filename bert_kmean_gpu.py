import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn import metrics

# 指定GPU设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)
# 加载预训练的BERT模型和分词器，并将它们移动到GPU上
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 示例短文本列表
# texts = ["This is an example sentence.",
#          "Another example sentence with different content.",
#          "Yet another example for demonstration purposes."]

data = pd.read_csv('Summary_mysql_csv_all_bugs-13.csv', lineterminator='\n')
df = DataFrame(data)
texts = []
for index, row in df.iterrows():
    texts.append(row['Summary'])

print("texts loaded")


# 对短文本进行分词和嵌入表示
tokenized_texts = [tokenizer.tokenize(text) for text in texts]
max_length = max(len(tokens) for tokens in tokenized_texts)
indexed_tokens = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
padded_tokens = [tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)) for tokens in indexed_tokens]
# tokens_tensor = torch.tensor(padded_tokens).to(device)

# 使用K-means聚类算法对嵌入向量进行聚类
num_clusters = 200  # 聚类数量
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# 在GPU上计算嵌入向量
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

# 创建数据加载器并将数据移动到GPU上
dataset = torch.tensor(padded_tokens).to(device)
dataloader = DataLoader(dataset, batch_size = 1024)
embeddings = []
# 在GPU上计算嵌入向量
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

# 将多个批次的嵌入向量拼接为一个数组
embeddings = np.concatenate(embeddings, axis=0)


# 将数据移回CPU进行聚类分析
cluster_labels = kmeans.fit_predict(embeddings)

# 保存结果到CSV文件
with open('clusters-gpu-13-200.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Text', 'Cluster'])
    for i, text in enumerate(texts):
        writer.writerow([text, cluster_labels[i]])
print("Results saved to clusters.csv file.")


# 计算轮廓系数
silhouette_score = metrics.silhouette_score(embeddings, cluster_labels)

# 计算内聚度和间隔度
cohesion = metrics.calinski_harabasz_score(embeddings, cluster_labels)
separation = metrics.davies_bouldin_score(embeddings, cluster_labels)

print("Silhouette Score:", silhouette_score)
print("Cohesion:", cohesion)
print("Separation:", separation)


