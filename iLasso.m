function [result AIC BIC] = iLasso(Series, lambda, krnl)
% Learning teporal dependency among irregular time series ussing Lasso (or its variants)
%
% INPUTS:
%       Series: an Nx1 cell array; one cell for each time series. Each cell
%               is a 2xT matrix. First row contains the values and the
%               second row contains SORTED time stamps. The first time
%               series is the target time series which is predicted.
%       lambda: The regularization parameter in Lasso
%       krnl:   Selects the kernel. Default is Gaussian. Available options
%               are Sinc (krnl = Sinc) and Inverse distance (krnl = Dist).
% OUTPUTS:
%       result: The NxL coefficient matrix.
%       AIC:    The AIC score
%       BIC:    The BIC score
% 
% Dependency: This code requires the GLMnet package to perform Lasso.
% For Details of the algorithm please refer to:
% M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)

% Parameters
L = 50;     % Length of studied lag
Dt = 0.5;   % \Delta t
SIG = 2;    % Kernel parameter. Here Gaussian Kernel Bandwidth
B = sum(Series{1}(2, :)<=(L*Dt));
N1 = size(Series{1}, 2);
P = length(Series);

% Build the matrix elements
Am = zeros(N1-B, P*L);
bm = zeros(N1-B, 1);

% Building the design matrix
for i = (B+1):N1
    bm(i-B) = Series{1}(1, i);
    for j = 1:P
        ti = (Series{1}(2, i) - L*Dt):Dt:(Series{1}(2, i)-Dt);
        ti = repmat(ti, length(Series{j}(2, :)), 1);
        tSelect = repmat(Series{j}(2, :)', 1, L);
        ySelect = repmat(Series{j}(1, :)', 1, L);
        K = exp(-((ti-tSelect).^2)/SIG);        % The Gaussian Kernel
        switch krnl
            case 'Sinc'     % The sinc Kernel
                Kp = sinc((ti-tSelect)/SIG);
            case 'Dist'     % The Dist Kernel
                Kp = SIG./((ti-tSelect).^2);
            otherwise
                Kp = K;
        end
        Am(i-B, ((j-1)*L+1):(j*L) ) = sum(ySelect.*Kp)./sum(Kp);
    end
end

% Solving Lasso using a solver; here the 'GLMnet' package
opt = glmnetSet;
opt.lambda = lambda;
opt.alpha = 1;
fit = glmnet(Am, bm, 'gaussian', opt);
w = fit.beta;

% Computing the BIC and AIC metrics
BIC = norm(Am*w-bm)^2-log(N1-B)*sum(w==0)/2;
AIC = norm(Am*w-bm)^2-2*sum(w==0)/2;

% Reformatting the output
result = zeros(P, L);
for i = 1:P
    result(i, :) = w((i-1)*L+1:i*L);
end