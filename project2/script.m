% load the data
load diabetes;
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];
%%% FILL CODE FOR PROBLEM 1 %%%
% linear regression without intercept
error_train = 0;
error_test = 0;
% linear regression with intercept
error_train_i = 0;
error_test_i = 0;

% calculated weights using linear regression
w_train = 0;
w_train_i = 0;

w_train = learnOLERegression(x_train,y_train);
w_train_i = learnOLERegression(x_train_i,y_train);

error_train = sqrt(sum((y_train' - (w_train' * x_train')).*(y_train' - (w_train' * x_train'))));
error_test = sqrt(sum((y_test' - (w_train' * x_test')).*(y_test' - (w_train' * x_test'))));
error_train_i = sqrt(sum((y_train' - (w_train_i' * x_train_i')).*(y_train' - (w_train_i' * x_train_i'))));
error_test_i = sqrt(sum((y_test' - (w_train_i' * x_test_i')).*(y_test' - (w_train_i' * x_test_i'))));

%%% END PROBLEM 1 CODE %%%

%%% FILL CODE FOR PROBLEM 2 %%%
% ridge regression using least squares - minimization
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);

w_ridge_train_i = 0;
w_ridge_test_i = 0;

for i = 1:length(lambdas)
    lambda = lambdas(i);
    % fill code here for prediction and computing errors
    w_ridge_train_i = learnRidgeRegression(x_train_i,y_train,lambda);
%   w_ridge_test_i = learnRidgeRegression(x_test_i,y_test,lambda);
    train_errors(i,1) = sqrt(sum((y_train' - (w_ridge_train_i' * x_train_i')).*(y_train' - (w_ridge_train_i' * x_train_i'))));
    test_errors(i,1) = sqrt(sum((y_test' - (w_ridge_train_i' * x_test_i')).*(y_test' - (w_ridge_train_i' * x_test_i'))));
end
[min_train_error, lambda_optimal_index] = min(test_errors);
lambda_optimal = lambdas(lambda_optimal_index);
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
%%% END PROBLEM 2 CODE %%%

%%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
initialWeights = zeros(65,1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);

% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, lambda);
    w_ridge_train_i = fmincg(objFunction, initialWeights, options);
% fill code here for prediction and computing errors
    train_errors(i,1) = sqrt(sum((y_train' - (w_ridge_train_i' * x_train_i')).*(y_train' - (w_ridge_train_i' * x_train_i'))));
    test_errors(i,1) = sqrt(sum((y_test' - (w_ridge_train_i' * x_test_i')).*(y_test' - (w_ridge_train_i' * x_test_i'))));
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
%%% END PROBLEM 3 CODE

%%% BEGIN  PROBLEM 4 CODE
% using variable number 3 only
x_train = x_train(:,3);
x_test = x_test(:,3);
train_errors = zeros(7,1);
test_errors = zeros(7,1);

% no regularization
lambda = 0;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w_non_linear = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    train_errors(d+1,1) = sqrt(sum((y_train' - (w_non_linear' * x_train_n')).*(y_train' - (w_non_linear' * x_train_n'))));
    test_errors(d+1,1) = sqrt(sum((y_test' - (w_non_linear' * x_test_n')).*(y_test' - (w_non_linear' * x_test_n'))));
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');

% optimal regularization
lambda = lambda_optimal; % from part 2
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w_non_linear = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    train_errors(d+1,1) = sqrt(sum((y_train' - (w_non_linear' * x_train_n')).*(y_train' - (w_non_linear' * x_train_n'))));
    test_errors(d+1,1) = sqrt(sum((y_test' - (w_non_linear' * x_test_n')).*(y_test' - (w_non_linear' * x_test_n'))));
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');

