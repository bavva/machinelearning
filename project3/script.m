clearvars;
clc;

[train_data, train_label, validation_data, validation_label, test_data, test_label] = preprocess();

save('dataset.mat', 'train_data', 'train_label', 'validation_data', 'validation_label', 'test_data', 'test_label');
load('dataset.mat');

n_class = 10;
T = zeros(size(train_label, 1), n_class);
for i = 1 : n_class
    T(:, i) = (train_label == i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic Regression with Gradient Descent*******************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = optimset('MaxIter', 200);
W = zeros(size(train_data, 2) + 1, n_class);
initialWeights = zeros(size(train_data, 2) + 1, 1);
for i = 1 : n_class
    objFunction = @(params) blrObjFunction(params, train_data, T(:, i));
    [w, ~] = fmincg(objFunction, initialWeights, options);
    W(:, i) = w;
end

predicted_label = blrPredict(W, train_data);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predicted_label == train_label)) * 100);
logistic_reg_train_acc = mean(double(predicted_label == train_label)) * 100;

predicted_label = blrPredict(W, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(predicted_label == validation_label)) * 100);
logistic_reg_val_acc = mean(double(predicted_label == validation_label)) * 100;

predicted_label = blrPredict(W, test_data);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predicted_label == test_label)) * 100);
logistic_reg_test_acc = mean(double(predicted_label == test_label)) * 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic Regression with Newton-Raphson method**************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (extra credits)
W = zeros(size(train_data, 2) + 1, n_class);
initialWeights = zeros(size(train_data, 2) + 1, 1);
n_iter = 5;
for i = 1 : n_class
    W(:, i) = blrNewtonRaphsonLearn(initialWeights, train_data, T(:, i), n_iter);
end

predicted_label = blrPredict(W, train_data);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predicted_label == train_label)) * 100);
lr_hessian_train_acc = mean(double(predicted_label == train_label)) * 100;

predicted_label = blrPredict(W, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(predicted_label == validation_label)) * 100);
lr_hessian_val_acc = mean(double(predicted_label == validation_label)) * 100;

predicted_label = blrPredict(W, test_data);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predicted_label == test_label)) * 100);
lr_hessian_test_acc = mean(double(predicted_label == test_label)) * 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiclass Logistic Regression with Gradient Descent *******
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (extra credits)
options = optimset('MaxIter', 200);
initialWeights = zeros((size(train_data, 2) + 1) * n_class, 1);

objFunction = @(params) mlrObjFunction(params, train_data, T);
[W, cost] = fmincg(objFunction, initialWeights, options);
W = reshape(W, size(train_data, 2) + 1, n_class);    

predicted_label = mlrPredict(W, train_data);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predicted_label == train_label)) * 100);
mlr_train_acc = mean(double(predicted_label == train_label)) * 100;

predicted_label = mlrPredict(W, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(predicted_label == validation_label)) * 100);
mlr_val_acc = mean(double(predicted_label == validation_label)) * 100;

predicted_label = mlrPredict(W, test_data);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predicted_label == test_label)) * 100);
mlr_test_acc = mean(double(predicted_label == test_label)) * 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiclass Logistic Regression with Newton-Raphson method **
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (extra credits)
% (un-comment this block of code to run)
% initialWeights = zeros((size(train_data, 2) + 1) * n_class, 1);
% n_iter = 5;
% [W] = mlrNewtonRaphsonLearn(initialWeights, train_data, T, n_iter);
% 
% predicted_label = mlrPredict(W, train_data);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(predicted_label == train_label)) * 100);
% 
% predicted_label = mlrPredict(W, validation_data);
% fprintf('\nValidation Set Accuracy: %f\n', mean(double(predicted_label == validation_label)) * 100);
% 
% predicted_label = mlrPredict(W, test_data);
% fprintf('\nTest Set Accuracy: %f\n', mean(double(predicted_label == test_label)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Support Vector Machine**************************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using linear kernel
model = svmtrain(train_label, train_data, '-t 0');
[~, accuracy, ~] = svmpredict(train_label, train_data, model);
fprintf('\nSVM linear kernel Training Set Accuracy: %f\n', accuracy);
svm_t0_train_acc = accuracy;

[~, accuracy, ~] = svmpredict(validation_label, validation_data, model);
fprintf('\nSVM linear kernel Validation Set Accuracy: %f\n', accuracy);
svm_t0_val_acc = accuracy;

[~, accuracy, ~] = svmpredict(test_label, test_data, model);
fprintf('\nSVM linear kernel Test Set Accuracy: %f\n', accuracy);
svm_t0_test_acc = accuracy;

% Using radial basis function with value of gamma setting to 1
model = svmtrain(train_label, train_data, '-t 2 -g 1');
[~, accuracy, ~] = svmpredict(train_label, train_data, model);
fprintf('\nSVM radial basis function with gamma 1 Training Set Accuracy: %f\n', accuracy);
svm_t2g1_train_acc = accuracy;

[~, accuracy, ~] = svmpredict(validation_label, validation_data, model);
fprintf('\nSVM radial basis function with gamma 1 Validation Set Accuracy: %f\n', accuracy);
svm_t2g1_val_acc = accuracy;

[~, accuracy, ~] = svmpredict(test_label, test_data, model);
fprintf('\nSVM radial basis function with gamma 1 Test Set Accuracy: %f\n', accuracy);
svm_t2g1_test_acc = accuracy;

% Using radial basis function with value of gamma setting to default
model = svmtrain(train_label, train_data, '-t 2');
[~, accuracy, ~] = svmpredict(train_label, train_data, model);
fprintf('\nSVM radial basis function with gamma default Training Set Accuracy: %f\n', accuracy);
svm_t2_train_acc = accuracy;

[~, accuracy, ~] = svmpredict(validation_label, validation_data, model);
fprintf('\nSVM radial basis function with gamma default Validation Set Accuracy: %f\n', accuracy);
svm_t2_val_acc = accuracy;

[~, accuracy, ~] = svmpredict(test_label, test_data, model);
fprintf('\nSVM radial basis function with gamma default Test Set Accuracy: %f\n', accuracy);
svm_t2_test_acc = accuracy;

% Using radial basis function with different values of C
costs = [1 10:10:100];
train_accuracies = zeros(length(costs),1);
validation_accuracies = zeros(length(costs),1);
test_accuracies = zeros(length(costs),1);

for i = 1:length(costs)
    cost = costs(i);
    libsvm_options = sprintf('-t 2 -c %d', cost);
    model = svmtrain(train_label, train_data, libsvm_options);
    
    [~, accuracy, ~] = svmpredict(train_label, train_data, model);
    train_accuracies(i, 1) = accuracy;
    [~, accuracy, ~] = svmpredict(validation_label, validation_data, model);
    validation_accuracies(i, 1) = accuracy;
    [~, accuracy, ~] = svmpredict(test_label, test_data, model);
    test_accuracies(i, 1) = accuracy;
end

save('results.mat', 'logistic_reg_train_acc', 'logistic_reg_val_acc', 'logistic_reg_test_acc', 'lr_hessian_train_acc', 'lr_hessian_val_acc', 'lr_hessian_test_acc', 'mlr_train_acc', 'mlr_val_acc', 'mlr_test_acc', 'svm_t0_train_acc', 'svm_t0_val_acc', 'svm_t0_test_acc', 'svm_t2g1_train_acc', 'svm_t2g1_val_acc', 'svm_t2g1_test_acc', 'svm_t2_train_acc', 'svm_t2_val_acc', 'svm_t2_test_acc', 'costs', 'train_accuracies', 'validation_accuracies', 'test_accuracies');

figure;
plot(costs', train_accuracies, costs', validation_accuracies, costs', test_accuracies);
legend('Training Accuracy', 'Validation Accuracy', 'Testing Accuracy');
xlabel Cost
ylabel Accuracy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
