clear 
clc
[X,Y]=meshgrid(-1:0.01:1);
Z=sin(pi.*X).*cos(pi.*Y);
ExData=Z(:)';               %期望拟合数据
x1=-1:0.01:1;               %产生拟合数据范围
x2=-1:0.01:1;
point = 201*201;              %拟合数据点个数
learn=0.2;                  %学习率
Weight1=rand(2,6);          %输入层到隐藏层权重    
Value1=rand(1,6);           %隐藏层阈值
Weight2=rand(6,1);          %隐藏层到输出层权重
Value2=rand(1,1);           %输出层阈值
InputHide=zeros(1,6);       %隐藏层输入
OutputHide=zeros(1,6);      %隐藏层输出
OutputLayer=0;              %输出层输出
Weight1d=zeros(2,6);        %输入层到隐藏层梯度项    
Value1d=zeros(1,6);         %隐藏层阈值梯度项  
Weight2d=zeros(6,1);        %输出层权重梯度项 
Value2d=zeros(1,1);         %隐藏层输出层阈值梯度项  
U=zeros(1,point);           %记录输出值
Error=zeros(1,point);       %误差
E=0;
Ee=0;                       %单组数据误差
i=0;

for x=1:201
    for y=1:201
        i=i+1;
            for time=1:200   
                %开始正向传播
                for j=1:6   
                    InputHide(j)=x1(x)*Weight1(1,j)+x2(y)*Weight1(2,j);    
                    OutputHide(j)=1/(1+exp(-(InputHide(j)-Value1(j))));
                end
                InputLayer=0;
                for j=1:6
                InputLayer=InputLayer+OutputHide(j)*Weight2(j,1);  
                end  
                if ExData(i)>=0 %由于sigmiod不具有映射负数范围的能力，所以根据期望值改变正负
                    OutputLayer=1/(1+exp(-(InputLayer-Value2)));      
                else
                    OutputLayer=-1/(1+exp(-(InputLayer-Value2)));
                end 
                Ee=((OutputLayer-ExData(i))^2)/2;                          %计算单组数据误差
               %反向传播 
                E=-(ExData(i)-OutputLayer)*OutputLayer*(1-OutputLayer);
                for j=1:6
                     Weight2d(j) = E*OutputHide(j);
                end   
                Value2d(1)= E;  
                for j=1:6
                    Weight1d(1,j) = E*OutputHide(j)*(1-OutputHide(j))*x1(x)*Weight2(j);
                end
                for j=1:6
                    Weight1d(2,j) = E*OutputHide(j)*(1-OutputHide(j))*x2(y)*Weight2(j);
                end
                for j=1:6
                    Value1d(j) = E*OutputHide(j)*(1-OutputHide(j));   
                end 
                %开始更新  
                for j=1:6
                    Weight2(j) = Weight2(j)-learn*Weight2d(j);
                end   
                    Value2 = Value2-learn*Value2d(1);       
                for j=1:6
                    Weight1(1,j) = Weight1(1,j)-learn*Weight1d(1,j);
                    Weight1(2,j) = Weight1(2,j)-learn*Weight1d(2,j);
                end   
                for j=1:6
                    Value1(j) = Value1(j)-learn*Value1d(j);
                end 
                %判断单点误差是否小于0.0001
                if Ee < 0.0001
                    Error(i) = Ee;
                break;
                end
            end  
    %记录输出值        
    U(i)=OutputLayer;
    if i==point
        i=0;
    end
    
    end       
end  

%绘制图像
figure(1)
[x,y]=meshgrid(-1:0.01:1);
subplot(2,2,1);
z=sin(pi.*x).*cos(pi.*y);
mesh(x,y,z);
title('目标函数的图像') 

subplot(2,2,2);
out=reshape(U,201,201);
mesh(x,y,out);
title('BP网络拟合后的图像')

subplot(2,2,[3,4]);
axis([0,42000,0,0.0001])
plot(Error);
title('BP网络拟合后与实际图像的误差')

%测试泛华效果
xx1=rand(1,100)*2-1;
xx2=rand(1,100)*2-1;
ZZ=zeros(1,100);
s1=zeros(1,100);
Os1=zeros(1,100);
EER=zeros(1,100);
for b=1:100
    ZZ(b)=sin(pi*xx1(b))*cos(pi*xx1(b));     
end
for b=1:100
     for j=1:6   
        s1(j)=xx1(b)*Weight1(1,j)+xx2(b)*Weight1(2,j);    
        Os1(j)=1/(1+exp(-s1(j)-Value1(j)));
     end
        OUt=0;
     for j=1:6
        OUt=OUt+Os1(j)*Weight2(j,1);  
     end            
     if ZZ(b)>=0 
       OUt=1/(1+exp(-(OUt-Value2)));      
         else
        OUt=-1/(1+exp(-(OUt-Value2)));
       end 
     EER(b)=((OUt-ZZ(b))^2)/2;  
end
figure(2)
plot(EER);
hold on
AVgEER=mean(EER);
A=ones(1,100)*AVgEER;
plot(1:100,A,'r')
title('BP网络泛化的误差与平均误差')   
       
       