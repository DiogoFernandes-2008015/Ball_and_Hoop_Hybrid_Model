% %Code for nonlinear grey model identification based on the matlab toolbox
% %Diogo Lopes Fernandes
% %The model must be defined on an auxliary function defined as
% %dx/dt=F(t,u,x)
% %y=H(t,u,x)
% %[dx, y]=model(t,x,u,p1,p2,...,pN,FileArgument)
% %dx -> derivative of the states of the model (F(t,x,u))
% %y -> output of the model (H(t,x,u))
% %p1,...,pN -> parameters to be estimated

clear all
close all
clc

%%      
%Load experimental data
DATA = load('seqDegrausRand.mat');
Ts = DATA.out.tout(2)-DATA.out.tout(1);
data = iddata(DATA.out.simout.omega__V_.Data/10,DATA.out.simout.cmd__pct_.Data,Ts,'Name','B&H - Random Steps');
data.InputName = 'Command Signal';
data.InputUnit = 'signal';
data.OutputName = 'Angular Velocity';
data.OutputUnit = 'rad/s';
%data.Domain = 'Time';
figure
idplot(data)

DATAval = load("seqDegrausFixo.mat");
Ts = DATA.out.tout(2)-DATA.out.tout(1);
dataval = iddata(DATAval.out.simout.omega__V_.Data/10,DATAval.out.simout.cmd__pct_.Data,'Name','B&H - Fixed Steps');
dataval.InputName = 'Command Signal';
dataval.InputUnit = 'signal';
dataval.OutputName = 'Angular Velocity';
dataval.OutputUnit = 'rad/s';
%dataval.Domain = 'Time';
figure
idplot(dataval)

%%
%Modelo Ball and Hoop Linear
%Construction o the idnlgrey object
Ny = 1; %number of outputs
Nx = 1; %number of states
Nu = 1; %number of inputs
Order = [Ny Nu Nx];

%Description of the parameters
Parameters(1).Name = 'J';%name of the parameter
Parameters(1).Unit = 'kgm^2';%Unit of the parameter
Parameters(1).Value =99*rand()+0.01;%initial estimative for the parameter
Parameters(1).Minimum = 0.001;%minimum value for the parameter
Parameters(1).Maximum = 100;%maximum value for the parameter
Parameters(1).Fixed = 0;
Parameters(2).Name = 'K';%name of the parameter
Parameters(2).Unit = 'V';%Unit of the parameter
Parameters(2).Value = 999*rand();%initial estimative for the parameter
Parameters(2).Minimum = 0.001;%minimum value for the parameter
Parameters(2).Maximum = 1000;%maximum value for the parameter
Parameters(2).Fixed = 0;
Parameters(3).Name = 'B';%name of the parameter
Parameters(3).Unit = 'Ns/rad';%Unit of the parameter
Parameters(3).Value = 999*rand();%initial estimative for the parameter
Parameters(3).Minimum = 0.001;%minimum value for the parameter
Parameters(3).Maximum = 1000;%maximum value for the parameter
Parameters(3).Fixed = 0;
%Initial Condition of the model
InitialStates = [0];

%Handle to the model function
func_handle = @bandh_linear;

nlgrmod = idnlgrey(func_handle, Order, Parameters, InitialStates);
nlgrmod.Ts = 0;
nlgrmod.InputName = 'Command Signal';
nlgrmod.InputUnit = 'signal';
nlgrmod.OutputName = 'Angular Velocity';
nlgrmod.OutputUnit = 'rad/s';

%nlgrmod

%%
%Estimation phase
%Configurations of the estimations algorithim
nlgropt = nlgreyestOptions;
nlgropt.Display = 'on';
nlgropt.SearchOptions.MaxIterations = 10000;
nlgropt.SearchOptions.FunctionTolerance = 1e-8;
nlgropt.SearchOptions.StepTolerance = 1e-8;


%Estimaton command
nlgrmod = nlgreyest(data, nlgrmod, nlgropt);

%Results
disp('Linear Results')
nlgrmod.Report.Fit
nlgrmod.Report.Parameters.ParVector
nlgrmod.Report.Termination

%%
%Results analysis
figure
subplot(2,2,1)
compare(data,nlgrmod)
title('Comparação dados de treinamento')
subplot(2,2,2)
resid(data,nlgrmod)
title('Testes de correlação dados de treinamento')
subplot(2,2,3)
compare(dataval,nlgrmod)
title('Comparação dados de validação')
subplot(2,2,4)
resid(dataval,nlgrmod)
title('Testes de correlação com dados de validação')
out=compare(data,nlgrmod);
disp('Treinamento')
r2 = mult_corr(data.OutputData,out.OutputData)
rmse = sqrt((data.OutputData-out.OutputData)'*(data.OutputData-out.OutputData)/length(data.OutputData))
mae = mean(abs(data.OutputData-out.OutputData))
disp('Validação')
out=compare(dataval,nlgrmod);
r2 = mult_corr(dataval.OutputData,out.OutputData)
rmse = sqrt((dataval.OutputData-out.OutputData)'*(dataval.OutputData-out.OutputData)/length(dataval.OutputData))
mae = mean(abs((dataval.OutputData-out.OutputData)))

%%
% %Saving the models
mod_bandh_lin = nlgrmod
% 
% %%
% %Modelo Ball and Hoop Linear with Stribeck Effect
% %Construction o the idnlgrey object
% Ny = 1; %number of outputs
% Nx = 1; %number of states
% Nu = 1; %number of inputs
% Order = [Ny Nu Nx];
% 
% %Description of the parameters
% Parameters(1).Name = 'J';%name of the parameter
% Parameters(1).Unit = 'kgm^2';%Unit of the parameter
% Parameters(1).Value =99*rand()+0.01;%initial estimative for the parameter
% Parameters(1).Minimum = 0.001;%minimum value for the parameter
% Parameters(1).Maximum = 100;%maximum value for the parameter
% Parameters(1).Fixed = 0;
% Parameters(2).Name = 'K';%name of the parameter
% Parameters(2).Unit = 'V';%Unit of the parameter
% Parameters(2).Value = 900*rand();%initial estimative for the parameter
% Parameters(2).Minimum = 0.001;%minimum value for the parameter
% Parameters(2).Maximum = 1000;%maximum value for the parameter
% Parameters(2).Fixed = 0;
% Parameters(3).Name = 'B';%name of the parameter
% Parameters(3).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(3).Value = 999*rand();%initial estimative for the parameter
% Parameters(3).Minimum = 0.001;%minimum value for the parameter
% Parameters(3).Maximum = 1000;%maximum value for the parameter
% Parameters(3).Fixed = 0;
% Parameters(4).Name = 'fc';%name of the parameter
% Parameters(4).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(4).Value = 999*rand();%initial estimative for the parameter
% Parameters(4).Minimum = 0.001;%minimum value for the parameter
% Parameters(4).Maximum = 1000;%maximum value for the parameter
% Parameters(4).Fixed = 0;
% Parameters(5).Name = 'fs';%name of the parameter
% Parameters(5).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(5).Value = 999*rand();%initial estimative for the parameter
% Parameters(5).Minimum = 0.001;%minimum value for the parameter
% Parameters(5).Maximum = 1000;%maximum value for the parameter
% Parameters(5).Fixed = 0;
% Parameters(6).Name = 'vs';%name of the parameter
% Parameters(6).Unit = 'exp';%Unit of the parameter
% Parameters(6).Value = 99*rand();%initial estimative for the parameter
% Parameters(6).Minimum = 0.001;%minimum value for the parameter
% Parameters(6).Maximum = 100;%maximum value for the parameter
% Parameters(6).Fixed = 0;
% %Initial Condition of the model
% InitialStates = [0];
% 
% %Handle to the model function
% func_handle = @bandh_linear_stribeck;
% 
% nlgrmod = idnlgrey(func_handle, Order, Parameters, InitialStates);
% nlgrmod.Ts = 0;
% nlgrmod.InputName = 'Command Signal';
% nlgrmod.InputUnit = 'signal';
% nlgrmod.OutputName = 'Angular Velocity';
% nlgrmod.OutputUnit = 'rad/s';
% 
% 
% %%
% %Estimation phase
% %Configurations of the estimations algorithim
% nlgropt = nlgreyestOptions;
% nlgropt.Display = 'on';
% nlgropt.SearchOptions.MaxIterations = 10000;
% nlgropt.SearchOptions.FunctionTolerance = 1e-8;
% nlgropt.SearchOptions.StepTolerance = 1e-8;
% 
% 
% 
% %Estimaton command
% nlgrmod = nlgreyest(data, nlgrmod, nlgropt);
% %Results
% disp('Linear with Stribeck')
% nlgrmod.Report.Fit
% nlgrmod.Report.Parameters.ParVector
% nlgrmod.Report.Termination
% 
% %%
% %Results analysis
% figure
% subplot(2,2,1)
% compare(data,nlgrmod)
% title('Comparação dados de treinamento')
% subplot(2,2,2)
% resid(data,nlgrmod)
% title('Testes de correlação dados de treinamento')
% subplot(2,2,3)
% compare(dataval,nlgrmod)
% title('Comparação dados de validação')
% subplot(2,2,4)
% resid(dataval,nlgrmod)
% title('Testes de correlação com dados de validação')
% out=compare(data,nlgrmod);
% disp('Treinamento')
% r2 = mult_corr(data.OutputData,out.OutputData)
% rmse = sqrt((data.OutputData-out.OutputData)'*(data.OutputData-out.OutputData)/length(data.OutputData))
% mae = mean(abs(data.OutputData-out.OutputData))
% disp('Validação')
% out=compare(dataval,nlgrmod);
% r2 = mult_corr(dataval.OutputData,out.OutputData)
% rmse = sqrt((dataval.OutputData-out.OutputData)'*(dataval.OutputData-out.OutputData)/length(dataval.OutputData))
% mae = mean(abs((dataval.OutputData-out.OutputData)))
% 
% %%
% %Saving the models
% mod_bandh_lin_stribeck = nlgrmod
% 
% %%
% %Modelo Ball and Hoop with LuGre Friction
% %Construction o the idnlgrey object
% Ny = 1; %number of outputs
% Nx = 2; %number of states
% Nu = 1; %number of inputs
% Order = [Ny Nu Nx];
% 
% %Description of the parameters
% Parameters(1).Name = 'J';%name of the parameter
% Parameters(1).Unit = 'kgm^2';%Unit of the parameter
% Parameters(1).Value =99*rand()+0.01;%initial estimative for the parameter
% Parameters(1).Minimum = 0.001;%minimum value for the parameter
% Parameters(1).Maximum = 100;%maximum value for the parameter
% Parameters(1).Fixed = 0;
% Parameters(2).Name = 'K';%name of the parameter
% Parameters(2).Unit = 'V';%Unit of the parameter
% Parameters(2).Value = 90*rand();%initial estimative for the parameter
% Parameters(2).Minimum = 0.001;%minimum value for the parameter
% Parameters(2).Maximum = 100;%maximum value for the parameter
% Parameters(2).Fixed = 0;
% Parameters(3).Name = 'B';%name of the parameter
% Parameters(3).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(3).Value = 99*rand();%initial estimative for the parameter
% Parameters(3).Minimum = 0.001;%minimum value for the parameter
% Parameters(3).Maximum = 100;%maximum value for the parameter
% Parameters(3).Fixed = 0;
% Parameters(4).Name = 'fc';%name of the parameter
% Parameters(4).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(4).Value = 99*rand();%initial estimative for the parameter
% Parameters(4).Minimum = 0.001;%minimum value for the parameter
% Parameters(4).Maximum = 100;%maximum value for the parameter
% Parameters(4).Fixed = 0;
% Parameters(5).Name = 'fs';%name of the parameter
% Parameters(5).Unit = 'Ns/rad';%Unit of the parameter
% Parameters(5).Value = 99*rand();%initial estimative for the parameter
% Parameters(5).Minimum = 0.001;%minimum value for the parameter
% Parameters(5).Maximum = 100;%maximum value for the parameter
% Parameters(5).Fixed = 0;
% Parameters(6).Name = 'vs';%name of the parameter
% Parameters(6).Unit = 'exp';%Unit of the parameter
% Parameters(6).Value = rand();%initial estimative for the parameter
% Parameters(6).Minimum = 0.0001;%minimum value for the parameter
% Parameters(6).Maximum = 1;%maximum value for the parameter
% Parameters(6).Fixed = 0;
% Parameters(7).Name = 's0';%name of the parameter
% Parameters(7).Unit = 'Ns/m';%Unit of the parameter
% Parameters(7).Value = 9*rand();%initial estimative for the parameter
% Parameters(7).Minimum = 0.001;%minimum value for the parameter
% Parameters(7).Maximum = 10;%maximum value for the parameter
% Parameters(7).Fixed = 0;
% Parameters(8).Name = 's1';%name of the parameter
% Parameters(8).Unit = 'Ns/m';%Unit of the parameter
% Parameters(8).Value = 9*rand();%initial estimative for the parameter
% Parameters(8).Minimum = 0.001;%minimum value for the parameter
% Parameters(8).Maximum = 10;%maximum value for the parameter
% Parameters(8).Fixed = 0;
% Parameters(9).Name = 's2';%name of the parameter
% Parameters(9).Unit = 'exp';%Unit of the parameter
% Parameters(9).Value = 9*rand();%initial estimative for the parameter
% Parameters(9).Minimum = 0.001;%minimum value for the parameter
% Parameters(9).Maximum = 10;%maximum value for the parameter
% Parameters(9).Fixed = 0;
% Parameters(10).Name = 'dv';%name of the parameter
% Parameters(10).Unit = 'exp';%Unit of the parameter
% Parameters(10).Value = 9*rand();%initial estimative for the parameter
% Parameters(10).Minimum = 0.001;%minimum value for the parameter
% Parameters(10).Maximum = 10;%maximum value for the parameter
% Parameters(10).Fixed = 0;
% %Initial Condition of the model
% InitialStates = [0;0];
% 
% %Handle to the model function
% func_handle = @bandh_lugre;
% 
% nlgrmod = idnlgrey(func_handle, Order, Parameters, InitialStates);
% nlgrmod.Ts = 0;
% nlgrmod.InputName = 'Command Signal';
% nlgrmod.InputUnit = 'signal';
% nlgrmod.OutputName = 'Angular Velocity';
% nlgrmod.OutputUnit = 'rad/s';
% 
% %%
% %Estimation phase
% %Configurations of the estimations algorithim
% nlgropt = nlgreyestOptions;
% nlgropt.Display = 'on';
% nlgropt.SearchOptions.MaxIterations = 10000;
% nlgropt.SearchOptions.FunctionTolerance = 1e-8;
% nlgropt.SearchOptions.StepTolerance = 1e-8;
% 
% 
% %Estimaton command
% nlgrmod = nlgreyest(data, nlgrmod, nlgropt)
% disp('Lugre Results')
% %Results
% nlgrmod.Report.Fit
% nlgrmod.Report.Parameters.ParVector
% nlgrmod.Report.Termination
% 
% %%
% %Results analysis
% figure
% subplot(2,2,1)
% compare(data,nlgrmod)
% title('Comparação dados de treinamento')
% subplot(2,2,2)
% resid(data,nlgrmod)
% title('Testes de correlação dados de treinamento')
% subplot(2,2,3)
% compare(dataval,nlgrmod)
% title('Comparação dados de validação')
% subplot(2,2,4)
% resid(dataval,nlgrmod)
% title('Testes de correlação com dados de validação')
% out=compare(data,nlgrmod);
% disp('Modelo Linear+LuGre')
% disp('Treinamento')
% r2 = mult_corr(data.OutputData,out.OutputData)
% rmse = sqrt((data.OutputData-out.OutputData)'*(data.OutputData-out.OutputData)/length(data.OutputData))
% mae = mean(abs(data.OutputData-out.OutputData))
% disp('Validação')
% out=compare(dataval,nlgrmod);
% r2 = mult_corr(dataval.OutputData,out.OutputData)
% rmse = sqrt((dataval.OutputData-out.OutputData)'*(dataval.OutputData-out.OutputData)/length(dataval.OutputData))
% mae = mean(abs((dataval.OutputData-out.OutputData)))
% 
% %%
% %Saving the models
% mod_bandh_lugre = nlgrmod
% %%
% %Saving the models
% save('modelos_grey_bandh','mod_bandh_lin','mod_bandh_lin_stribeck','mod_bandh_lugre')