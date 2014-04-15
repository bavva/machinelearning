function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1

% Extracting dimensions from inputs

[N D] = size(X);

w = zeros(D,1);

[y_rows y_cols] = size(y);

if (not(y_rows == N))
    sprintf('Training data and label should have the same number of rows!!');
else
    if (not(y_cols == 1))
        sprintf('Training label should a N*1 column matrix!!');
    else      
        
        % I_D = eye(D+1);
        X_transpose = transpose(X);

        % w_ridge = inverse(lambda*N*I_D + transpose_X*X)*transpose_X*y;
        
        w = mtimes(inv(lambda*N*eye(D) + mtimes(X_transpose,X)),mtimes(X_transpose,y));
    end
end


end
