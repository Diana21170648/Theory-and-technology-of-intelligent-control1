clc
clear
close all
%% 绘制函数图像 
x1=-1:0.05:1; %生成步长为0.05,范围[-1,1]的数组训练数据x共41个
x2=-1:0.05:1;
[X1,X2]=meshgrid(x1,x2); %生成41*41矩阵
yd=sin(pi*X1).*cos(pi*X2);
subplot(2,1,1);
surf(X1,X2,yd);
title('目标函数输出期望图形');
Input=2;
HideLayer=5; % 神经元的个数
Output=1;
%% 权值及阈值初始化
w1=rands(HideLayer,Input); %隐含层的权值和阈值
b1=rands(HideLayer,1);
w2=rands(Output,HideLayer); %输出层的权值和阈值
b2=rands(Output,1);
% dw1=zeros(HideLayer,Input);
% dw2=zeros(Output,HideLayer);
error = 0.01; %误差阈值
alpha=0.05; %学习速率
mu=0.5; %S函数系数
M=size(x1,2);
for m1=1:M
for m2=1:M
y=sin(pi*x1(m1))*cos(pi*x2(m2));
%% 激活前向传播
for k=1:50
for i=1:HideLayer %隐层的输出 
s1(i)=x1(m1)*w1(i,1)+x2(m2)*w1(i,2)+b1(i,1);
HOut(i)=(1+exp(-mu.*s1(i))).^-1;
end
s2=0; %输出层
for j=1:HideLayer %求解输出层的输入
Out_In(j)=HOut(j)*w2(1,j);
s2=s2+Out_In(j);
end
s2=s2+b2;
if y>=0 %输出层的输出
dy =(1+exp(-mu.*s2)).^-1;
else
dy = -(1+exp(-mu.*s2)).^-1;
end
%% 反向，更新权重
e0 = y - dy;
e = e0^2/2;
D_Out = -e0 * (1 - dy) * dy;
for k = 1:HideLayer
Delta_HideLayer(k) = D_Out * w2(1,k) * (1 - HOut(k)) * HOut(k);
end
for k=1:HideLayer
w2(1,k)=w2(1,k)-alpha*D_Out*HOut(k);
end
b2(1,1)=b2(1,1)-alpha*D_Out;
for i=1:HideLayer
for j=1:Input
if j==1 
w1(i,j)=alpha*Delta_HideLayer(i)*x1(m1);
else
w1(i,j)=alpha*Delta_HideLayer(i)*x2(m2);
end 
% b1(i,1)=b1(i,1)+xite*Delta_HideLayer(i);
end
b1(i,1)=b1(i,1)+alpha*Delta_HideLayer(i);
end
%% 判断停止迭代
if e < error
break;
end
end
y1(m2,m1) = dy;
e1(m2,m1) = e;
end
end
subplot(2,1,2);
surf(X1,X2,y1);
title('BP神经网络逼近sin(pi*x1)*cos(pi*x2)图像');