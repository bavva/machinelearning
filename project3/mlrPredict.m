function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = horzcat(ones(size(X, 1), 1), X);

% Calculate A
A = X * W;

expA = exp(A);
sumExpA = sum(expA, 2);
exapandedSumExpA = zeros(size(A));
for i = 1:size(A, 2)
    exapandedSumExpA(:, i) = sumExpA;
end
Y = expA ./ exapandedSumExpA;

[~, label] = max(Y, [], 2);

end

