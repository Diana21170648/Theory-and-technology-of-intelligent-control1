%=========================
%清空环境变量
clc
clear
close all

%=========================
%BP网络层数、每层节点数设置
In=2;
Hid=6;
Out=1;

%给定初始权值和阈值
Win = rands(Hid,In);%隐层权值
Theta1= rands(Hid,1);%隐层阈值
Wout= rands(Out,Hid);%输出层权值
Theta2= rands(Out,1);%输出层阈值
%期望误差
Eexp = 0.00001;
%步长
Alpha = 0.6;
%样本数
L = 201;

%=========================
%确定自变量x1和x2的取值范围和取值间隔
x1=-1:0.01:1;
x2=-1:0.01:1;
%利用meshgrid指令生成“格点”矩阵
[X1,X2]=meshgrid(x1,x2);
%生成原函数的三维图形图像并作保留
f1=sin(pi*X1).*cos(pi*X2);
figure;
mesh(X1,X2,f1);
title('函数f=sin(pi*x1)*cos(pi*x2)三维图形')
hold on;

%=========================
%递推过程
for c1 = 1:L
    for c2 = 1:L
        f2 = sin(pi*x1(c1))*cos(pi*x2(c2));
        for loop=1:50
            for i=1:Hid
                %隐层的输出
                Hidout(i) =  logsig(x1(c1)*Win(i,1)+x2(c2)*Win(i,2)+Theta1(i,1));
            end
            s=0;
            for j=1:Hid
                %输出层的输入
                Outin(j)=Hidout(j)*Wout(1,j);
                s=s+Outin(j);
            end
            s=s+Theta2
            %输出层的输出
            if f2>=0
                Outout = logsig(s);
            elseif f2<0
                Outout = -logsig(s);
            end
            %误差
            E0 = f2 - Outout;
            
            E = E0^2/2;
            Delta_Out= -E0*(1-Outout)*Outout;
            for d = 1:Hid
                Delta_Hid(d) = Delta_Out * Wout(1,d) * (1 - Hidout(d)) * Hidout(d);                
            end
            for d = 1:Hid
                Wout(1,d)=Wout(1,d)-Alpha*Delta_Out*Hidout(d);
            end
            Theta2(1,1)=Theta2(1,1)-Alpha*Delta_Out;
            for i=1:Hid
                for j=1:In
                    if j==1
                       Win(i,j)=Alpha*Delta_Hid(i)*x1(c1);
                    else
                       Win(i,j)=Alpha*Delta_Hid(i)*x2(c2);
                    end
                end
                Theta1(i,1)=Theta1(i,1)+Alpha*Delta_Hid(i);        
            end
            %停止迭代条件
            if E < Eexp
                break;
            end
        end
        f3(c2,c1) = Outout;
        E1(c2,c1) = E;
    end
end

%=========================
%生成BP神经网络逼近原函数的三维图形
figure;
mesh(x1,x2,f3);
title('BP神经网络逼近原函数的三维图形');
%生成误差变化图
figure;
mesh(x1,x2,E1);
title('误差变化图');