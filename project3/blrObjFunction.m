function [error, error_grad] = blrObjFunction(w, X, t)
% blrObjFunction computes 2-class Logistic Regression error function and
% its gradient.
%
% Input:
% w: the weight vector of size (D + 1) x 1 
% X: the data matrix of size N x D
% t: the label vector of size N x 1 where each entry can be either 0 or 1
%    representing the label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size (D+1) x 1 representing the gradient of
%             error function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add ones in the beginning
X = horzcat(ones(size(X, 1), 1), X);

% Calculate Y
Y = sigmoid (X * w);

% Calculate error
error = sum((t .* log(Y)) + ((1 - t) .* log(1 - Y))) .* (-1);

% Calculate error gradiance
error_grad = X' * (Y - t);


end
