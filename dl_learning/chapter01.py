import torch

# 张量表示一个数值组成的数组，这个数组可能有多个维度。
x = torch.arange(12)
print(x)

print(x.shape)
"""
shape访问张量的形状和元素的数量
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
"""

print(x.numel())
"""元素总数"""

x = x.reshape(3, 4)
print(x)
"""
改变张量形状而不改变元素数量和元素值
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
        """

y = torch.zeros((2, 3, 4))
z = torch.ones((2, 3, 4))
print(y)
print(y.shape)
print(z)
"""
使用全0，全1，随机数初始化张量
->2个三行四列的数组
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])

torch.Size([2, 3, 4])

tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
"""

x = torch.tensor([[4, 5, 6], [7, 8, 9]])
print(x)
print(x.shape)
x = torch.tensor([[[4, 5, 6], [7, 8, 9]]])
print(x)
print(x.shape)
"""
tensor([[4, 5, 6],
        [7, 8, 9]])
torch.Size([2, 3])

tensor([[[4, 5, 6],
         [7, 8, 9]]])
torch.Size([1, 2, 3])
        """

# 常见的数学运算符都可以被升级为按元素运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **是求幂运算
print(torch.exp(x))  # 指数运算
"""
tensor([ 3.,  4.,  6., 10.]) 
tensor([-1.,  0.,  2.,  6.]) 
tensor([ 2.,  4.,  8., 16.]) 
tensor([0.5000, 1.0000, 2.0000, 4.0000]) 
tensor([ 1.,  4., 16., 64.])

tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
"""

# 把多个张量连结在一起

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))  # 序列数组
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)
print(torch.cat((X, Y), dim=0))  # dim=0表示在行上连结(最外面的中括号
print(torch.cat((X, Y), dim=1))  # dim=1表示在列上连结(也就是第几个中括号
"""
X:tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
Y:tensor([[2., 1., 4., 3.],
        [1., 2., 3., 4.],
        [4., 3., 2., 1.]])
dim=0
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  3.,  2.,  1.]])        
dim=1
tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])
"""

# 逻辑运算符构建二维张量--按元素进行
print(X == Y)
"""
tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
        """

print(x.sum())  # 张量所有元素的和
"""
tensor(15.)
    """

# 广播机制（broadcasting mechanism）
a = torch.arange(3).reshape((3, 1))  # 三行一列
b = torch.arange(2).reshape((1, 2))  # 一行两列
print(a, b)
"""
##维度一样，都是二维,而且同一维要是整数倍才可以广播
tensor([[0],
        [1],
        [2]]) 
tensor([[0, 1]])
a->3x2  通过复制
b->3x2
        """
print(a + b)
"""
tensor([[0, 1],
        [1, 2],
        [2, 3]])
"""

# 元素的访问
print(X[-1])  # 最后一行
print(X[1:3])  # （从0开始数）第一行和第二行
"""
tensor([ 8.,  9., 10., 11.])
tensor([[ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
        """
# 写入
X[1, 2] = 9
print(X)
"""
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
"""

# 为多个元素同时赋值
X[0:2, :] = 12  # 0-1行所有的列都赋值为12
print(X)
X[-1, 1:3] = 0  # 最后一行的1-2列赋值为0
print(X)
"""
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  0.,  0., 11.]])
        """

# 运行一些操作可能会导致为新结果分配内存
before = id(Y)  # id()函数提供了内存中引用对象的确切地址,类似于指针
Y = Y + X
print(id(Y) == before)
"""
False
    """
# 执行原地操作
Z = torch.zeros_like(Y)  # 创建一个形状和Y相同，但是元素为0的张量
print('id(Z):', id(Z))
Z[:] = X + Y  # 原地操作
print('id(Z):', id(Z))
before = id(X)
X += Y  # 原地操作
"""
+=相当于有[:]的操作,给矩阵中的元素重新赋值，
但是x=x+y是整个矩阵的操作，重新分配内存

python拷贝，深拷贝，浅拷贝
赋值拷贝（Assignment Copy）:
    这不是真正的拷贝，而是创建了一个新的引用指向同一个对象。
    例子：a = [1, 2, 3] 然后 b = a，此时 b 和 a 都指向同一个列表对象。
浅拷贝（Shallow Copy）:
    通过创建一个新的对象，其内容是原对象中对象的引用的拷贝。
    浅拷贝会拷贝容器对象，但不会拷贝容器内部的对象。
    例子：使用 list.copy() 方法或 copy 模块的 copy() 函数。
    如果原对象中包含对其他对象的引用，这些引用所指向的对象不会被拷贝。
深拷贝（Deep Copy）:
    创建一个新的对象，并且递归地拷贝原对象中所引用的对象。
    深拷贝会拷贝容器对象以及容器内部的对象。
    例子：使用 copy 模块的 deepcopy() 函数。
    深拷贝可以确保新对象与原始对象完全独立，修改新对象不会影响原始对象。
"""
print(id(X) == before)
"""
    id(Z): 2215400931072
    id(Z): 2215400931072
    True
    """
print("各种拷贝对比")
idx = id(X)
A = X
print(id(A) == idx)  # True
B = X.clone()
print(id(B) == idx)  # False
C = X[:]
print(id(C) == idx)  # False

# 转化为Numpy张量
A = X.numpy()  # Numpy的多元数组
B = torch.tensor(A)  # torch的tensor
print(type(A), type(B))
"""
<class 'numpy.ndarray'> <class 'torch.Tensor'>
"""
a = torch.tensor([1, 2, 3])
print(a)
b = a.numpy()
print(b)
a += 1
print(a)
print(b)

# 将大小为1的张量转化为python张量
a = torch.tensor([3.5])
print(a)
print(
    a.item())  # Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
print(float(a))
print(int(a))
"""
tensor([3.5000])
3.5
3.5
3
    """

# 数据预处理
import os
import pandas as pd  # 用于读取csv数据集

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本 NA表示未知
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)
"""
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
"""
# 为了处理缺失数据，插值、删除
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # index location
# data_4*3
# inputs_4*2 所有的行，第0，1列
# outputs_4*1 所有的行，第2列
inputs = inputs.fillna(inputs.mean())  # 用均值填充缺失数据
print(inputs)
"""
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
    """

# 对于非数值域的，把缺失值变成另一类 用不同数值填充
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
"""
NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
#pave=[1,0],NA=[0,1]
"""

# 转化为张量
X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(Y)
"""
tensor([[3., 1., 0.],
        [2., 0., 1.],
        [4., 0., 1.],
        [3., 0., 1.]], dtype=torch.float64)
tensor([127500, 106000, 178100, 140000])
    """

# reshape 和 view 的区别
a = torch.arange(12)
b = a.reshape(3, 4)
b[:] = 2
print(a)
"""
tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """