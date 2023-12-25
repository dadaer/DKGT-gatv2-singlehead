import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# 假设我们有一些训练数据
# 每个训练示例包括(head entity, relation, tail entity)的索引
class TripletsDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, shape, output_channels, dropout_rate=0.1):
        super(ConvE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim  # Total dimensions for entity and relation embeddings
        self.shape = shape  # Shape to reshape the embeddings before convolution
        self.output_channels = output_channels  # Number of output channels after convolution

        # Define entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(3, 3), stride=1, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(output_channels * shape[0] * shape[1] * 2, embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Batch normalization
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, heads, relations):
        # Get embeddings
        head_embeddings = self.entity_embeddings(heads).view(-1, 1, self.shape[0], self.shape[1])
        relation_embeddings = self.relation_embeddings(relations).view(-1, 1, self.shape[0], self.shape[1])

        # Concatenate head and relation embeddings along the width
        x = torch.cat([head_embeddings, relation_embeddings], 2)  # resulting shape: (batch_size, 1, 2*shape[0], shape[1])

        # Apply 2D convolution
        x = self.bn0(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Reshape for fully connected layer
        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(x))

        # Apply batch normalization
        x = self.bn2(x)

        # Predict
        x = F.relu(x)

        # Score against all entities
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))

        return x


# 创建一些示例数据
num_entities = 1000
num_relations = 1000
num_samples = 10000
triplets = torch.randint(low=0, high=num_entities, size=(num_samples, 3))  # 随机生成一些训练数据

# 创建数据集和数据加载器
dataset = TripletsDataset(triplets)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 创建ConvE模型
model = ConvE(num_entities, num_relations, embedding_dim=200, shape=(10, 20), output_channels=32, dropout_rate=0.3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 进入训练模式
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]

        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        scores = model(heads, relations)

        # 计算损失
        loss = criterion(scores, tails)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 100 == 99:  # 每100批次打印一次
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Training finished.')
