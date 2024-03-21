'''
Descripttion: 
Author: 
Date: 2024-02-21 23:58:55
LastEditTime: 2024-02-22 19:39:46
'''
import numpy as np

import prepare_for_training

## 随机梯度下降: 根据矩阵计算

class LinearRegression: 
    ## label 有监督任务
    
    ## 1. 对数据进行预处理操作
    ## 2， 先得到所有特征个数
    ## 3. 初始化参数矩阵
    
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True) :
       
        (data_processed, features_mean, features_deviation) = prepare_for_training(data,polynomial_degree = 0 ,sinusoid_degree = 0,normalize_data = 0)
        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))
        
        
    ## 当前迭代次数 和 学习率
    ## 返回损失值
    ## 训练模块 执行梯度下降
    def train(self,alpha,num_iteration = 500):
        cost_history = self.gradient_descent(alpha,num_iteration)
        return self.theta, cost_history
        
    ## 实际迭代模块 会迭代nu_iterations
    def gradient_descent(self,alpha,num_iteration):
        cost_history = []
        
        for _ in range(num_iteration):
            ## 进行参数更新 
            self.gradient_step(alpha=alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
            
        return cost_history

               

    ## 梯度下降:参数更新计算方法,注意是矩阵运算
    def gradient_step(self,alpha):
        num_examples= self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        # 参差
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha *(1/num_examples)*(np.dot(delta.T ,self.data)).T
    
    def cost_function(self,data,labels):
        # 损失计算方法
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)
        return cost[0][0]
        
    
    @staticmethod    
    def hypothesis(data,theta):
        ## 预测值
        predictions = np.dot(data,theta)
        return predictions
        
    
    def get_cost(self,data,labels):
       data_processed =  prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]
       # 得到损失 
       return self.cost_function(data_processed,labels)
   
   
    def predict(self,data):
        # 用训练的参数模型 与预测得到的
        data_processed =  prepare_for_training
        (data,self.polynomial_degree,self.sinusoid_degree,self.normalize_data)[0]

        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        