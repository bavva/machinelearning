function [error, error_grad] = regressionObjVal(w, X, y, lambda)

N = size(X, 1);
error = ((sum((y' - (w' * X')) .* (y' - (w' * X')))) ./ N) + (lambda .* (w' * w));
error_grad = (lambda .* N .* w - (X' * (y - (X * w)))) .* 2 ./ N;

end
