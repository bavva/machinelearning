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

% calculate testing data size
test_rows = 0;
data_columns = size(test0, 2);
for i = 0:9
    test_matrix = eval(strcat('test', num2str(i)));
    test_rows = test_rows + size(test_matrix, 1);
end

% allocate space for test data and test label and load them
test_data = zeros(test_rows, data_columns);
test_label = zeros(test_rows, 1);
test_total_rows = 0;
for i = 0:9
    test_matrix = eval(strcat('test', num2str(i)));
    test_rows = size(test_matrix, 1);
    
    test_data(test_total_rows+1:test_total_rows+test_rows, :) = test_matrix;
    test_label(test_total_rows+1:test_total_rows+test_rows, :) = i;
    
    test_total_rows = test_total_rows + test_rows;
end

% calculate training data and validation data size
train_rows = 0;
validation_rows = 0;
for i = 0:9
    train_matrix = eval(strcat('train', num2str(i)));
    
    to_rows = size(train_matrix, 1);
    tr_rows = ceil(to_rows * (5.0/ 6.0));
    va_rows = to_rows - tr_rows;
    train_rows = train_rows + tr_rows;
    validation_rows = validation_rows + va_rows;
end

% loading train data and validation data
train_data = zeros(train_rows, data_columns);
train_label = zeros(train_rows, 1);
validation_data = zeros(validation_rows, data_columns);
validation_label = zeros(validation_rows, 1);
train_total_rows = 0;
validation_total_rows = 0;
for i = 0:9
    train_matrix = eval(strcat('train', num2str(i)));
    
    to_rows = size(train_matrix, 1);
    tr_rows = ceil(to_rows * (5.0/ 6.0));
    va_rows = to_rows - tr_rows;
    
    train_data(train_total_rows+1:train_total_rows+tr_rows, :) = train_matrix(1:1:tr_rows, :);
    validation_data(validation_total_rows+1:validation_total_rows+va_rows, :) = train_matrix(tr_rows+1:1:end, :);
    
    train_label(train_total_rows+1:train_total_rows+tr_rows) = i;
    validation_label(validation_total_rows+1:validation_total_rows+va_rows) = i;
    
    train_total_rows = train_total_rows + tr_rows;
    validation_total_rows = validation_total_rows + va_rows;
end

% converting to double
train_data = double(train_data);
validation_data = double(validation_data);
test_data = double(test_data);

% normalizing
train_data = mat2gray(train_data);
validation_data = mat2gray(validation_data);
test_data = mat2gray(test_data);

% feature selection
minvalvector = min(train_data, [], 1);
maxvalvector = max(train_data, [], 1);
for i = data_columns:-1:1
    maxval = maxvalvector(i);
    minval = minvalvector(i);
    if (maxval - minval < 0.0001)
        train_data(:, i) = [];
        validation_data(:, i) = [];
        test_data(:, i) = [];
    end
end

end

