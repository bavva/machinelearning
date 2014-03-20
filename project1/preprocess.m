function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of testing set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

load('mnist_all.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loading test data
test_data = [];
test_label = [];
for i = 1:10
    test_matrix = eval(strcat('test', num2str(i - 1)));
    test_data = vertcat(test_data, test_matrix);
    [rows, columns] = size(test_matrix);
    test_label = vertcat(test_label, [zeros(rows,(i-1)) ones(rows, 1) zeros(rows, 9-(i-1))]);
end

% loading train data and validation data
train_data = [];
train_label = [];
validation_data = [];
validation_label = [];

for i = 1:10
    train_matrix = eval(strcat('train', num2str(i - 1)));
    [rows, columns] = size(train_matrix);
    train_rows = ceil(rows * (5.0/ 6.0));
    validation_rows = rows - train_rows;
    train_data = vertcat(train_data, train_matrix(1:1:train_rows, :));
    validation_data = vertcat(validation_data, train_matrix(train_rows+1:1:end, :));
    train_label = vertcat(train_label, [zeros(train_rows,(i-1)) ones(train_rows,1) zeros(train_rows, 9-(i-1))]);
    validation_label = vertcat(validation_label, [zeros(validation_rows,(i-1)) ones(validation_rows,1) zeros(validation_rows, 9-(i-1))]);
end

% converting to double and normalizing
train_data = double(train_data);
train_data = mat2gray(train_data);
validation_data = double(validation_data);
validation_data = mat2gray(validation_data);
test_data = double(test_data);
test_data = mat2gray(test_data);

% feature selection
[rows, columns] = size(train_data);
minvalvector = min(train_data, [], 1);
maxvalvector = max(train_data, [], 1);
for i = columns:-1:1
    maxval = maxvalvector(i);
    minval = minvalvector(i);
    if (maxval - minval < 0.0001)
        train_data(:, i) = [];
        validation_data(:, i) = [];
        test_data(:, i) = [];
    end
end

end

