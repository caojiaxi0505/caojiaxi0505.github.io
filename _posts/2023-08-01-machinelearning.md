---
layout: post
author: caojiaxi
title: 机器学习笔记
type: study
---

# 机器学习笔记

## 机器学习简介

机器学习（machine learning，ML），主要研究如何使机器模拟人类学习活动

## 常见的机器学习算法

决策树算法，朴素贝叶斯算法，支持向量机算法，随机森林算法，神经网络算法，Boosting与Bagging算法，关联规则算法，期望最大化算法

## 支持向量机（SVM）

### SVM简介

SVM本质是一种分类器

### 线性可分

$D_0$和$D_1$是$n$维欧式空间中的两个点集，如果存在$n$维向量$w$和实数$b$，使得$D_0$中的点$x_i$满足$w x_i + b \gt 0$，对于$D_1$中的点$x_j$满足$w x_j + b \lt 0$，则称$D_0$和$D_1$线性可分

### 超平面与最大间隔超平面（最佳超平面）

将$D_0$和$D_1$完全分开的$w x + b$就是一个超平面，最大间隔超平面满足以下两个条件：

1. 两类样本分别在该超平面的两侧
2. 两侧距离超平面最近的点到超平面的距离被最大化

### 支持向量

样本中距离超平面最近的点，称为支持向量

### SVM最优化

SVM解决的问题就是找到最大间隔超平面

一般超平面方程：$w^T x + b = 0$

点$x = (x_1 , x_2 , \cdots , x_n)$到超平面的距离为${\lvert w^T x + b \rvert \over \lVert w \rVert}$，其中$\lVert w \rVert = \sqrt{w_1^2 + w_2^2 + \cdots + w_n^2}$

支持向量到超平面的距离我们这里规定为$d$，有

$\begin{cases}{w^T x + b \over \lVert w \rVert} \ge d , y = 1 \\\ {w^T x + b \over \lVert w \rVert} \le -d , y = -1\end{cases}$

稍作转化，有

$\begin{cases}{w^T x + b \over \lVert w \rVert d} \ge 1 , y = 1 \\\ {w^T x + b \over \lVert w \rVert d} \le -1 , y = -1\end{cases}$

由于$\lVert w \rVert d$为正数，我们可以令其为1（对函数的优化过程无影响），有

$\begin{cases}w^T x + b \ge 1 , y = 1 \\\ w^T x + b \le -1 , y = -1\end{cases}$

简写为

$y(w^T x + b) \ge 1$

支持向量到超平面的距离可以写为

$d = {\lvert w^T x + b \rvert \over \lVert w \rVert} = {y (w^T x + b) \over \lVert w \rVert}$

我们求最大间隔超平面，即求解以下问题

$max~2 * d$，其中$d =  {y (w^T x + b) \over \lVert w \rVert}$

带入条件$y(w^T x + b) = 1$，问题转化为求解$max~ {2 \over \lVert w \rVert}$

进一步转化为$min~ {\lVert w \rVert \over 2}$

最终转化为$min~ { {\lVert w \rVert}^2 \over 2}$

### 不等式约束优化（Lagrange乘数法的扩展）

问题描述如下：

$min~ { {\lVert w \rVert}^2 \over 2}$

$s.t.~g_i(w) = 1 - y_i(w^T x_i + b) \le 0$

引入松弛变量$a_i^2$，将约束转化为$h_i(w) = g_i(w) + a_i^2 = 0$

Lagrange函数为

$L(w,\lambda,a) = f(w) + \sum_{i=1}^n \lambda_i h_i(w) = f(w) + \sum_{i=1}^n \lambda_i[g_i(w) + a_i^2]$

令

$\begin{cases}{\partial L \over \partial w_i} = {\partial f \over \partial w_i} + {\sum_{i=1}^n \lambda_i {\partial g_i \over \partial w_i}} = 0 \\\ {\partial L \over \partial a_i} = 2 \lambda a_i = 0 \\\ {\partial L \over \partial \lambda_i} = g_i(w) + a_i^2 = 0 \\\ \lambda_i \ge 0\end{cases}$

得到最优解的必要条件

$2 \lambda_i a_i = 0$可以分为两种情况

1. $\lambda_i = 0, a_i \ne 0$

    $g_i(w) \lt 0$

2. $\lambda_i \ne 0, a_i = 0$

    $g_i(w) = 0$

可以转化为

$\begin{cases}{\partial L \over \partial w_i} = {\partial f \over \partial w_i} + {\sum_{i=1}^n \lambda_i {\partial g_i \over \partial w_i}} = 0 \\\ g_i(w) \lambda_i  = 0 \\\ g_i(w) \le 0 \\\ \lambda_i \ge 0\end{cases}$

称之为不等式约束的KKT条件

$min~ { {\lVert w \rVert}^2 \over 2} \iff min~L(w,\lambda,a)$

又有含$a$项$\sum_{i=1}^n \lambda_i a_i^2 \ge 0$

问题转化为

$min~L(w,\lambda)$

假设${ {\lVert w \rVert}^2 \over 2}$的最小值为$p$，$L(w,\lambda)$中$\sum_{i=1}^n \lambda_i g_i(w) \le 0$，因此取得最小值时，有$L(w,\lambda) \le p$，令$L(w,\lambda) \rightarrow p$以求使${ {\lVert w \rVert}^2 \over 2}$最小对应的$\lambda$，即求$\max\limits_\lambda L(w,\lambda)$

最终的最优化问题为

$\min\limits_{w} \max\limits_{\lambda}~L(w,\lambda)$

$s.t.~\lambda_i \ge 0$

### 强对偶关系

$\min\limits_{w} \max\limits_{\lambda}~L(w,\lambda)$

$s.t.~\lambda_i \ge 0$

与

$\max\limits_{\lambda} \min\limits_{w}~L(w,\lambda)$

$s.t.~\lambda_i \ge 0$

称为对偶问题

如果满足$\max\limits_{\lambda} \min\limits_{w}~f = \min\limits_{w} \max\limits_{\lambda}~f$，则称$f$具有强对偶性

$f$是凸优化问题$\iff$$f$具有强对偶性$\iff$KKT条件

### SVM优化求解步骤（重点）

step1. 构造Lagrange函数

$\min\limits_{w,b} \max\limits_{\lambda}~L(w,b,\lambda)= \min\limits_{w,b} \max\limits_{\lambda}~{ {\lVert w \rVert}^2 \over 2} + \sum_{i=1}^n \lambda_i[1-y_i(w^T x_i + b)]$

$s.t.~\lambda_i \ge 0$

step2. 强对偶条件转化

$\max\limits_{\lambda} \min\limits_{w,b} ~L(w,b,\lambda)$

对$w,b$求偏导数

$\begin{cases}{\partial L \over \partial w} = w + \sum_{i=1}^n \lambda_i x_i y_i = 0 \\\ {\partial L \over \partial b} = -\sum_{i=1}^n \lambda_i y_i  = 0 \end{cases}$

得到
$\begin{cases}\sum_{i=1}^n \lambda_i x_i y_i = w \\\ \sum_{i=1}^n \lambda_i y_i = 0 \end{cases}$

带入Lagrange函数

$\min\limits_{w,b}L(w,b,\lambda)=\sum_{i=1}^n\lambda_i - {1 \over 2}\sum_{i=1}^n\sum_{j=1}^n \lambda_i \lambda_j y_i y_j(x_i x_j)$

$s.t.~\sum_{i=1}^n\lambda_i y_i = 0, \lambda_i \ge 0$

step3. 求$\max\limits_{\lambda}~\sum_{i=1}^n\lambda_i - {1 \over 2}\sum_{i=1}^n\sum_{j=1}^n \lambda_i \lambda_j y_i y_j(x_i x_j)$

这是一个二次规划算法，使用SMO算法求解

step4. $w = \sum_{i=1}^n \lambda_i x_i y_i$求得$w$

step5. 带入支持向量$y_s(w x_s + b) = 1$，求得$b$

带入多个支持向量求均值即为最终的$b$
