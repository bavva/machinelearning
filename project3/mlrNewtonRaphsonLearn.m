function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = horzcat(ones(size(X, 1), 1), X);
W = initial_W;

for i = 1:n_iter
    Wold = W;
    
    % Calculate A
    A = X * Wold;
    expA = exp(A);
    sumExpA = sum(expA, 2);
    exapandedSumExpA = zeros(size(A));
    for i = 1:size(A, 2)
        exapandedSumExpA(:, i) = sumExpA;
    end
    Y = expA ./ exapandedSumExpA;
    Egrad = X' * (Y - T);

    
    W = Wold - (Egrad / H);
end

end

