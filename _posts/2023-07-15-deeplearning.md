---
layout: post
author: caojiaxi
title: 深度学习笔记
type: study
---

# 深度学习笔记

## torch杂记

### torch.utils.data.Dataloader

```python
torch.utils.data.Dataloader(dataset, batch_size, shuffle, num_workers)
```

### torchvision.dataset.MNIST

```python
torchvision.dataset.MNIST(root, train, transform, download)
```

### torchvision.transforms.Conpose

```python
torchvision.transforms.Conpose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])
```

### torchvision.transforms.ToTensor

```python
torchvision.transforms.ToTensor(image)
```

### torch.nn.Conv2d

> input_channel_num：输入通道数
>
> output_channel_num：输出通道数
>
> kernel_size：卷积核边长
>
> padding：填充

```python
torch.nn.Conv2d(<input_channel_num>, <output_channel_num>, kernel_size=<>, padding=<>)
```

### torch.nn.AvgPool2d

```python
torch.nn.AvgPool2d(kernel_size=<>, padding=<>)
```

### torch.nn.Linear

```python
torch.nn.Linear(<input_dim>, <output_dim>)
```

### torch.nn.sigmoid

```python
torch.nn.sigmoid()
```

### torch.nn.CrossEntropyloss

```python
criterion = torch.nn.CrossEntropyloss()
```

### torch.optim.SGD

```python
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```




---

## 项目代码结构

### 较复杂的项目

project-name

- data/
- checkpoints/
- logs/
- utils/
    - utils.py
    - metric.py
- models/
    - model.py
- configs/
    - config.yaml
- datasets/
    - data_loader.py
- main.py

### 较简单的项目

project-name

- data/
- checkpoints/
- logs/
- utils/
    - utils.py
    - metrics.py
- model.py
- config.yaml
- data_loader.py
- main.py

---

## 一些常见的源

### pip清华源

```shell
pip install <package_name> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### conda清华源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

---



## 其他知识点

### yaml文件读写

```python
import yaml

with open('./config.yml', 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
```

### data_loader.py写法

data_loader需定义两个类，一个是dataset类，一个是dataloader类

#### 自定义dataset类

> 继承torch.utils.data.Dataset

下面的例子，图片存储在"./images/"下，图片的名字与对应label均从dict.txt读取

```python
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path = './images/'
        with open('dict.txt', 'r', encoding='utf-8') as f:
            self.label_dict = eval(f.read())

    def __getitem__(self, index):
        label = list(self.label_dict.values())[index - 1]
        img_id = list(self.label_dict.keys())[index - 1]
        img_path = self.path + str(img_id) + ".jpg"
        img = np.array(Image.open(img_path))
        return img, label

    def __len__(self):
        return len(self.label_dict)
```

#### 创建dataloader

```python
import torch

dataset = MyDataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
for i in range(epoch):
    for index, (img, label) in enumerate(dataloader):
        pass
```

### model.py写法

#### 自定义net类

```python
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
    	x = torch.nn.functional.relu(self.fc1(x))
    	x = torch.nn.functional.relu(self.fc2(x))
    	x = self.fc3(x)
    	return torch.nn.functional.log_softmax(x)
    
net = Net()
```

#### 自定义优化器

> 随机梯度下降优化器

```python
optimizer =	 torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
```

#### 自定义损失函数

```python
criterion = nn.NLLLoss() # 负对数似然损失
```

#### 训练过程

```python
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        # 将数据大小从 (batch_size, 1, 28, 28) 变为 (batch_size, 28*28)
        data = data.view(-1, 28*28)
        optimizer.zero_grad()
        target_hat = net(data)
        loss = criterion(target_hat, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
```

### num_workers注意事项

> 当num_workers>0时，要将主函数内容写在下面代码中：

```python
if __name__ == '__main__':
```

### flatten注意事项

> 展平时如果不希望维度降低，需设定start_dim=1

```python
x = torch.flatten(x, start_dim=1)
```

### Net注意实现

> Net的forward函数要return x