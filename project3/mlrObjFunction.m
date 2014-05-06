function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function


W = reshape(W, size(X, 2) + 1, size(T, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add ones in the beginning
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
error_grad_pre_reshape = X' * (Y - T);

error = sum(sum(T .* log(Y))) .* (-1);
error_grad = reshape(error_grad_pre_reshape, size(error_grad_pre_reshape, 1) * size(error_grad_pre_reshape, 2), 1);

end
