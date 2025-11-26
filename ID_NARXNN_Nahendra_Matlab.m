%%
clear all
clc
close all

%%
DATA = load('seqDegrausRand.mat');
Ts = DATA.out.tout(2)-DATA.out.tout(1);
data = iddata(DATA.out.simout.omega__V_.Data/10,DATA.out.simout.cmd__pct_.Data,Ts,'Name','B&H - Random Steps');
data.InputName = 'Command Signal';
data.InputUnit = 'signal';
data.OutputName = 'Angular Velocity';
data.OutputUnit = 'rad/s';
%data.Domain = 'Time';
%figure
%idplot(data)

DATAval = load("seqDegrausFixo.mat");
Ts = DATA.out.tout(2)-DATA.out.tout(1); 
dataval = iddata(DATAval.out.simout.omega__V_.Data/10,DATAval.out.simout.cmd__pct_.Data,'Name','B&H - Fixed Steps');
dataval.InputName = 'Command Signal';
dataval.InputUnit = 'signal';
dataval.OutputName = 'Angular Velocity';
dataval.OutputUnit = 'rad/s';


% Verification dataset
ue = data.InputData;
ut = dataval.InputData;
yt = dataval.OutputData;
ye = data.OutputData;

%%
%Plotting of training and validation data
figure()
subplot(2,2,1)
plot(ue)
title('Input - Estimation Dataset')
grid
subplot(2,2,2)
plot(ut)
grid
title('Input - Validation Dataset')
subplot(2,2,3)
plot(ye)
grid
title('Output - Estimation Dataset')
subplot(2,2,4)
plot(yt)
grid
title('Output - Validation Dataset')
%%
%Construction of the regression matrix
nu = 5; %order of inputs
ny = 5; %order of outputs
Phie = matreg(ye,ue,ny,nu);
Phit = matreg(yt,ut,ny,nu);

%%
% %Construction of the Regression Neural Network Structure 
% nneur = 25
% nlayers = 3
% Net = fitrnet(Phie,ye(ny+1:end),"LayerSizes",nneur*ones(1,nlayers),"Activations","relu","Verbose",1,"LayerWeightsInitializer","glorot","IterationLimit",1e6,"GradientTolerance",1e-6,"LossTolerance",1e-6,"StepTolerance",1e-4)
%%
%Construction of a Deep Neural Network
nneur = 32;
layers= [sequenceInputLayer(nu+ny)
          fullyConnectedLayer(nneur)
          reluLayer()
          fullyConnectedLayer(nneur)
          reluLayer()
          fullyConnectedLayer(1)]
options = trainingOptions("adam", ...
    MaxEpochs=30000, ...
    SequencePaddingDirection="left", ...
    Shuffle="never", ...
    Plots="training-progress", ...
    Verbose=1, ...
    MiniBatchSize=length(Phie), ...
    Metrics='rmse', ...
    ExecutionEnvironment="cpu", ...
    OutputNetwork='auto')
Net = trainnet(Phie,ye(ny+1:end),layers,"mse",options);

%%
%Free Run Simulations
y_hat_e_FR = freeRun(Net,ye, ue, ny, nu);
%y_hat_OSA = predict(Net,Phie)
y_hat_v_FR = freeRun(Net,yt, ut, ny, nu);
%y_hat_v_OSA = predict(Net,Phit)

%%
%Plotting free run simulations results
figure()
hold on
plot(ye(ny+1:end),'LineStyle','-','Color','black')
plot(y_hat_e_FR,'LineStyle','-','Color','green','LineWidth',2)
hold off
figure()
hold on
plot(yt(ny+1:end),'LineStyle','-','Color','black')
plot(y_hat_v_FR,'LineStyle','-','Color','green','LineWidth',2)
hold off
%%
figure()
minY = min(min(yt), min(y_hat_v_FR));
maxY = max(max(yt), max(y_hat_v_FR));
scatter(yt(ny+1:end), y_hat_v_FR, 'red', 'DisplayName', 'Prediction');
hold on;
plot([minY, maxY], [minY, maxY], 'k', 'LineWidth', 2, 'DisplayName', 'Perfect model');
xlabel('Real');
ylabel('Prediction');
grid on;
legend;
hold off 