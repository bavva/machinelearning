function w = learnOLERegression(X,y)

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1


[N D] = size(X);
[y_rows y_columns] = size(y);

w = zeros(N,1);

if (not(y_rows == N))
    sprintf('X and y should have the same number of rows!!');
else
    if (not(y_columns == 1))
        sprintf('y should be a vector!!');
    else
        
        X_transpose = transpose(X);
        
        % w_MLE = inverse(transpose(X) * X) * transpose(X) * y;
        
        w = mtimes(inv(mtimes(X_transpose,X)) , mtimes(X_transpose,y));
        
    end
end

end