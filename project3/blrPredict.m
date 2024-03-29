function [label] = blrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10. Each column is the weight
%    vector of a Logistic Regression classifier.
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add ones in the beginning
X = horzcat(ones(size(X, 1), 1), X);

% Calculate Y(prediction) for all weights. Y will be N x 10
Y = sigmoid(X * W);

% In each row, which ever index has high value, that is our label for that
% row
[~, label] = max(Y, [], 2);

end

