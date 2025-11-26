%Modelo Ball and Hoop
%Código para estimação de modelos baseados em função de transferência
clear all
close all
clc

%%
%Carregamento dos dados
DATA = load('seqDegrausFixo.mat')
Ts = DATA.out.tout(2)-DATA.out.tout(1)
data = iddata(DATA.out.simout.omega__V_.Data,DATA.out.simout.cmd__pct_.Data,Ts,'Name','B&H - Fixed Steps');
data.InputName = 'Command Signal';
data.InputUnit = 'signal';
data.OutputName = 'Angular Velocity';
data.OutputUnit = 'rad/s';
data.Domain = 'Time';
figure
idplot(data)

DATAval = load("seqDegrausRand.mat");
Ts = DATA.out.tout(2)-DATA.out.tout(1);
dataval = iddata(DATAval.out.simout.omega__V_.Data,DATAval.out.simout.cmd__pct_.Data,'Name','B&H - Random Steps');
dataval.InputName = 'Command Signal';
dataval.InputUnit = 'signal';
dataval.OutputName = 'Angular Velocity';
dataval.OutputUnit = 'rad/s';
dataval.Domain = 'Time';
figure
idplot(dataval)

%%
%Estimação

Options = tfestOptions;
Options.InitialCondition = 'estimate';

tf11 =tfest(data,1,1,Options)
tf21 =tfest(data,2,1,Options)
tf22 =tfest(data,2,2,Options)
tf31 =tfest(data,3,1,Options)
tf32 =tfest(data,3,2,Options)
tf33 =tfest(data,3,3,Options)
tf41 =tfest(data,4,1,Options)
tf42 =tfest(data,4,2,Options)
tf43 =tfest(data,4,3,Options)
tf44 =tfest(data,4,4,Options)
tf51 =tfest(data,5,1,Options)
tf52 =tfest(data,5,2,Options)
tf53 =tfest(data,5,3,Options)
tf54 =tfest(data,5,4,Options)
tf55 =tfest(data,5,5,Options)

save('modelostf.mat','tf11','tf21','tf22','tf31','tf32',"tf33","tf55",'tf54','tf53','tf52','tf51','tf41','tf42','tf43','tf44')
%%
clc
figure(1)
subplot(2,2,1)
compare(data,tf33)
title('Comparação dados de treinamento')
subplot(2,2,2)
resid(data,tf33)
title('Testes de correlação dados de treinamento')
subplot(2,2,3)
compare(dataval,tf33)
title('Comparação dados de validação')
subplot(2,2,4)
resid(dataval,tf33)
title('Testes de correlação com dados de validação')
out=compare(data,tf33);
disp('Treinamento')
r2 = mult_corr(data.OutputData,out.OutputData)
rmse = sqrt((data.OutputData-out.OutputData)'*(data.OutputData-out.OutputData)/length(data.OutputData))
mae = mean(abs(data.OutputData-out.OutputData))
figure(2)
plot_xcorrel(data.OutputData-out.OutputData,data.InputData)
disp('Validação')
out=compare(dataval,tf33);
r2 = mult_corr(dataval.OutputData,out.OutputData)
rmse = sqrt((dataval.OutputData-out.OutputData)'*(dataval.OutputData-out.OutputData)/length(dataval.OutputData))
mae = mean(abs((dataval.OutputData-out.OutputData)))
figure(3)
plot_xcorrel(dataval.OutputData-out.OutputData,dataval.InputData)