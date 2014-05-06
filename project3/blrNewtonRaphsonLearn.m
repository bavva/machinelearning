function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = horzcat(ones(size(X, 1), 1), X);
w = initial_w;

for i = 1:n_iter
    wold = w;
    
    Y = sigmoid(X * wold);
    R = diag(sparse(Y .* (1 - Y)));
    H = X' * R * X;
    Egrad = X' * (Y - t);
    
    w = wold - (pinv(H) * Egrad);
end

end
