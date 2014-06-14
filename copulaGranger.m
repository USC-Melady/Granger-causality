function cause = copulaGranger(series)
% This is the code for computing Granger-Copula as described in the following paper:
% 	Yan Liu, Mohammad Taha Bahadori, and Hongfei Li, "Sparse-GEV: Sparse Latent Space Model for Multivariate Extreme Value Time Series Modeling", ICML 2012
% INPUT: 'series': N by T matrix
% OUTPUT: 'cause': The N by N Granger causal coefficients (sum of L coefficients)
% Dependency: lassoGranger (which depends on GLMnet package).
% The code performs automatic tuning of Lambda, using the metric that is defined in lassoGranger function.
% The user should provide the 'P' parameter value which specifies the number of lags used in Granger causality analysis.

P = 2;  % Number of Lags
% Setting the range of Lambda
nLam = 6; 
lambda = logspace(-3, 2, nLam);

T = size(series, 2);
N = size(series, 1);

delta = 1/(4*(T^(1/4))*sqrt(pi*log(T)));
mSeries = 0*series;
for i = 1:N
    mSeries(i, :) = map(series(i, :), delta);
end
mSeries = norminv(mSeries, 0, 1);


pError = 0*lambda;
causeTemp = zeros(N, nLam);
cause = zeros(N, N);
fprintf('Node #: %5d', 0);
for in = 1:N
    index = [in, 1:(in-1), (in+1):N];
    for i = 1:nLam
        [~, causeTemp(:, i), pError(i)] = lassoGranger(mSeries(index, :), P, lambda(i), 'l');
    end
    
    [~, id] = min(pError);
    index = [2:in, 1, (in+1):N];
    cause(:, in) = causeTemp(index, id);
    fprintf('%c%c%c%c%c%c', 8,8,8,8,8,8);
    fprintf('%5d ', in);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = map(Seri, delta)
out = 0*Seri;
for i = 1:length(Seri)
    out(i) = sum(Seri < Seri(i))/length(Seri);
end

out( out < delta) = delta;
out( out > 1-delta) = 1-delta;
end