% Run all
clc
clear all
addpath('Path to /glmnet_matlab/')

%% Generate a simple synthetic dataset
N = 20;     % # of time series
T = 100;    % length of time series
sig = 0.2;
genSynth(N, T, sig);

%% Run Lasso-Granger
lambda = 1e-2;
L = 1;      % Only one lag for analysis
load synth.mat
cause = zeros(N, N);
for in = 1:N
    index = [in, 1:(in-1), (in+1):N];
    [~, temp] = lassoGranger(series(index, :), L, lambda, 'l');
    cause(in, :) = temp([2:in, 1, (in+1):N])';
    fprintf('%c%c%c%c%c%c', 8,8,8,8,8,8);
    fprintf('%5d ', in);
end

%% Visualized the Matrix
subplot(1, 2, 1)
showMatrix(A)
title('Ground Truth')

subplot(1, 2, 2)
showMatrix(cause)
title('Inferred Causality')