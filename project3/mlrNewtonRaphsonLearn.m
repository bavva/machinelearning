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

W = reshape(initial_W, size(X, 2) + 1, size(T, 2));
X = horzcat(ones(size(X, 1), 1), X);

for temp = 1:n_iter
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
    FlatEgrad = reshape(Egrad, size(Egrad, 1) * size(Egrad, 2), 1);
    
    k = size(T, 2);
    KX = repmat(X, k, k);
    FlatY = reshape(Y, size(Y, 1) * size(Y, 2), 1);
    FlatYProduct = FlatY .* (1 - FlatY);
    FlatYProductDiag = diag(sparse(FlatYProduct));
    bigH = KX' * FlatYProductDiag * KX;
    
    bigDelta = pinv(bigH) * FlatEgrad;
    delta = reshape(bigDelta, size(Wold, 1), size(Wold, 2));
    
    W = Wold - delta;
end

end

