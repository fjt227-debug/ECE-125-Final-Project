%% Manual DNN (Sigmoid, Sigmoid, ReLU) with Mini-batch GD for Age->Salary regression
clear; clc; rng(1);  % for reproducibility
close all

%% Hyperparameters & architecture
inputSize = 5;
hiddenSize1 = 16; %%first hidden layer for Part F change from 8, 16, then 32
hiddenSize2 = 16; %%second hidden layer for Part F change from 8, 16, then 32
hiddenSize3 = 16; %%third hidden layer for Part F change from 8, 16, then 32
hiddenSize4 = 16; %%fourth hidden layer for Part F change from 8, 16, then 32
outputSize = 1;

learningRate = 0.01;
epochs = 2000;

load('traingDATA.mat','X','Y');
[numSamples,~]=size(X);

%% Split data: 80% train, 20% test
nTrain = floor(0.80*numSamples);

Xtrain = X(1:nTrain,:);
Ytrain = Y(1:nTrain,:);

Xtest = X(nTrain+1:end,:);
Ytest = Y(nTrain+1:end,:);

%% Normalize (z-score) using training set stats
muX = mean(Xtrain); sX=std(Xtrain);
muY = mean(Ytrain); sY=std(Ytrain);

Xtrain_n = (Xtrain-muX)./sX;
Ytrain_n = (Ytrain-muY)./sY;

Xtest_n = (Xtest-muX)./sX;
Ytest_n = (Ytest-muY)./sY;

frac = 1; %%For Part D change 100, 80, 60 percent of the training data 
nTrain_case = round(frac*nTrain);
idx_keep = randperm(nTrain,nTrain_case);   
Xtrain_case = Xtrain_n(idx_keep,:);
Ytrain_case = Ytrain_n(idx_keep,:);
nTrain = nTrain_case;

%% Activation functions and derivatives
%%For Part E Uncomment the differnt options to test ReLu, tanh, and sigmoid
%%Option 1: ReLU
act = @(x)max(0,x);
dact = @(x)double(x>0);
actName = 'ReLu';

%%Option 2: tanh
%act = @(x)tanh(x);
%dact = @(x)1-tanh(x).^2;
%actName = 'tanh';

%%Option 3: sigmoid
%act = @(x)1./(1+exp(-x));
%dact = @(x)act(x).*(1-act(x));
%actName = 'sigmoid';

%% Weight initialization 
W1 = randn(inputSize,hiddenSize1)*sqrt(1/inputSize);
b1 = zeros(1,hiddenSize1);

W2 = randn(hiddenSize1,hiddenSize2)*sqrt(1/hiddenSize1);
b2 = zeros(1,hiddenSize2);

W3 = randn(hiddenSize2,hiddenSize3)*sqrt(2/hiddenSize2);
b3 = zeros(1,hiddenSize3);

W4 = randn(hiddenSize3,hiddenSize4)*sqrt(2/hiddenSize3);
b4 = zeros(1,hiddenSize4);

W5 = randn(hiddenSize4,outputSize)*sqrt(1/hiddenSize4);
b5 = zeros(1,outputSize);

%% Activation functions and derivatives
sigmoid = @(x)1./(1+exp(-x));
sigmoid_deriv = @(s)s.*(1-s); %%input should be sigmoid(z) not z

relu = @(x)max(0,x);
relu_deriv = @(x)double(x>0); %%derivative based on pre-activation z

%% Training setup
trainingLoss = zeros(epochs,1); 
for epoch = 1:epochs
    
    %% Shuffle training samples each epoch 
    idx_epoch = randperm(nTrain); %%nTrain is defined above
    Xb = Xtrain_case(idx_epoch,:); %%shuffled inputs
    Yb = Ytrain_case(idx_epoch,:); %%shuffled targets
    B = size(Xb,1);

    z1 = Xb*W1+repmat(b1,B,1);%%[B X h1]
    a1 = act(z1); 

    z2 = a1*W2+repmat(b2,B,1);%%[B X h2]
    a2 = act(z2);

    z3 = a2*W3+repmat(b3,B,1);%%[B X h3]
    a3 = act(z3);

    z4 = a3*W4+repmat(b4,B,1);%%[B X h4]
    a4 = act(z4);

    z5 = a4*W5+repmat(b5,B,1);%%[B X 1]
    ypred = z5;

    %% Loss (MSE on training)
    err  = Yb-ypred; %%[B x 1]
    loss = mean(err.^2); %%scalar
    trainingLoss(epoch) = loss;
    
    %% Backpropagation 
    dL_dy = -2*err/B; %%[B x 1]

    dL_dW5 = a4'*dL_dy; %%[h4 x 1]
    dL_db5 = sum(dL_dy, 1); %%[1 x 1]

    %% Hidden layer 4 (W4, b4)
    dL_da4 = dL_dy*(W5'); %%[B x h4]
    dL_dz4 = dL_da4.*dact(z4); %%[B x h4] = delta_4
    dL_dW4 = a3'*dL_dz4; %%[h3 x h4]
    dL_db4 = sum(dL_dz4, 1); %%[1 x h4]
    
    %% Hidden layer 3 (W3, b3)
    dL_da3 = dL_dz4*(W4'); %%[B x h3]
    dL_dz3 = dL_da3.*dact(z3); %%[B x h3] = delta_3
    dL_dW3 = a2'*dL_dz3; %%[h2 x h3]
    dL_db3 = sum(dL_dz3, 1); %%[1 x h3]
    
    %% Hidden layer 2 (W2, b2)
    dL_da2 = dL_dz3*(W3'); %%[B x h2]
    dL_dz2 = dL_da2.*dact(z2); %%[B x h2] = delta_2
    dL_dW2 = a1'*dL_dz2; %%[h1 x h2]
    dL_db2 = sum(dL_dz2, 1); %%[1 x h2]

    %% Hidden layer 1 (W1, b1)
    dL_da1 = dL_dz2*(W2'); %%[B x h1]
    dL_dz1 = dL_da1.*dact(z1); %%[B x h1] = delta_1
    dL_dW1 = Xb'*dL_dz1; %%[inputSize x h1]
    dL_db1 = sum(dL_dz1,1); %%[1 x h1]
    
    %% Parameter updates
    W5 = W5-learningRate*dL_dW5;
    b5 = b5-learningRate*dL_db5;

    W4 = W4-learningRate*dL_dW4;
    b4 = b4-learningRate*dL_db4;
    
    W3 = W3-learningRate*dL_dW3;
    b3 = b3-learningRate*dL_db3;
    
    W2 = W2-learningRate*dL_dW2;
    b2 = b2-learningRate*dL_db2;
    
    W1 = W1-learningRate*dL_dW1;
    b1 = b1-learningRate*dL_db1;
        
    if mod(epoch, 25) == 0 || epoch==1
        fprintf('Epoch %3d / %3d  TrainMSE = %.6f\n', ...
            epoch, epochs, trainingLoss(epoch));
    end
end

%% Figure 1: Plot training MSE vs Epoch
figure;
plot(1:epochs, trainingLoss, 'LineWidth', 2); 
xlabel('Epoch'); ylabel('MSE (normalized units)'); 
title('Training MSE (normalized)'); grid on;

%% Evaluate on Test Set (vectorized)
Z1_tr = Xtrain_case*W1+repmat(b1,size(Xtrain_case,1),1);
A1_tr = act(Z1_tr);

Z2_tr = A1_tr*W2+repmat(b2,size(A1_tr,1),1);
A2_tr = act(Z2_tr);

Z3_tr = A2_tr*W3+repmat(b3,size(A2_tr,1),1);
A3_tr = act(Z3_tr);

Z4_tr = A3_tr*W4+repmat(b4,size(A3_tr,1),1);
A4_tr = act(Z4_tr);

Z5_tr = A4_tr*W5+repmat(b5,size(A4_tr,1),1);
Ytrain_pred_n = Z5_tr;

%% Denormalize for interpretability
Ytrain_pred = Ytrain_pred_n.*sY + muY;
Ytrain_true = Ytrain(idx_keep);               

%% Metrics on denormalized values
MSE_train  = mean((Ytrain_true-Ytrain_pred).^2);
RMSE_train = sqrt(MSE_train);
SSE_train  = sum((Ytrain_true-Ytrain_pred).^2);
SST_train  = sum((Ytrain_true-mean(Ytrain_true)).^2);
R2_train   = 1-SSE_train/SST_train;
NMSE_train = MSE_train/var(Ytrain_true);

%% Evaluate on Test Set (vectorized)
Z1 = Xtest_n*W1 + repmat(b1,size(Xtest_n,1),1);
A1 = act(Z1);
Z2 = A1*W2 + repmat(b2,size(A1,1),1);
A2 = act(Z2);
Z3 = A2*W3 + repmat(b3,size(A2,1),1);
A3 = act(Z3);
Z4 = A3*W4+repmat(b4,size(A3,1),1);
A4 = act(Z4);
Z5 = A4*W5+repmat(b5,size(A4,1),1);
Ytest_pred_n = Z5;

%% Denormalize for interpretability
Ytest_pred = Ytest_pred_n.*sY+muY;
Ytest_true = Ytest;         

%% Test metrics (MSE, RMSE, R^2, NMSE)
MSE_test  = mean((Ytest_true-Ytest_pred).^2);
RMSE_test = sqrt(MSE_test);
SSE_test  = sum((Ytest_true-Ytest_pred).^2);
SST_test  = sum((Ytest_true-mean(Ytest_true)).^2);
R2_test   = 1-SSE_test/SST_test;
NMSE_test = MSE_test/var(Ytest_true);

fprintf('\nTest performance (denormalized):\n');
fprintf('MSE  = %.10f, RMSE = %.6f, R^2 = %.6f, NMSE = %.6f\n', ...
        MSE_train, RMSE_train, R2_train, NMSE_train);

%% Forward pass for all N samples
X_all_n = (X-muX)./sX;

Z1_all = X_all_n*W1+repmat(b1,numSamples,1);
A1_all = act(Z1_all);

Z2_all = A1_all*W2+repmat(b2,numSamples,1);
A2_all = act(Z2_all);

Z3_all = A2_all*W3+repmat(b3,numSamples,1);
A3_all = act(Z3_all);

Z4_all = A3_all*W4+repmat(b4,numSamples,1);
A4_all = act(Z4_all);

Z5_all = A4_all*W5+repmat(b5,numSamples,1);
Y_all_pred_n = Z5_all;
Y_all_pred = Y_all_pred_n.*sY+muY;

%% Choose a range 
idx_plot = 50:2000;   %%adjust for a different window

%% Figure 2: Actual vs Predicted in transient region
figure; 
plot(idx_plot, Y(idx_plot), 'b', 'LineWidth', 1.5); 
hold on; plot(idx_plot, Y_all_pred(idx_plot), 'r--', 'LineWidth', 1.5); 
xlabel('Sample Time S'); ylabel('Output y'); 
legend('Actual y_k','Predicted \itŷ_k'); 
title('Actual vs Predicted output (transient region)'); 
grid on;

