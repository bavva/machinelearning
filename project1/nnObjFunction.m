function [obj_val obj_grad] = nnObjFunction(params, n_input, n_hidden, ...
                                    n_class, training_data,...
                                    training_label, lambda)
% nnObjFunction computes the value of objective function (negative log 
%   likelihood error function with regularization) given the parameters 
%   of Neural Networks, thetraining data, their corresponding training 
%   labels and lambda - regularization hyper-parameter.

% Input:
% params: vector of weights of 2 matrices w1 (weights of connections from
%     input layer to hidden layer) and w2 (weights of connections from
%     hidden layer to output layer) where all of the weights are contained
%     in a single vector.
% n_input: number of node in input layer (not include the bias node)
% n_hidden: number of node in hidden layer (not include the bias node)
% n_class: number of node in output layer (number of classes in
%     classification problem
% training_data: matrix of training data. Each row of this matrix
%     represents the feature vector of a particular image
% training_label: the vector of truth label of training images. Each entry
%     in the vector represents the truth label of its corresponding image.
% lambda: regularization hyper-parameter. This value is used for fixing the
%     overfitting problem.
       
% Output: 
% obj_val: a scalar value representing value of error function
% obj_grad: a SINGLE vector of gradient value of error function
% NOTE: how to compute obj_grad
% Use backpropagation algorithm to compute the gradient of error function
% for each weights in weight matrices.
% Suppose the gradient of w1 is 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape 'params' vector into 2 matrices of weight w1 and w2
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit i in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in hidden 
%     layer to unit i in output layer.
w1 = reshape(params(1:n_hidden * (n_input + 1)), ...
                 n_hidden, (n_input + 1));

w2 = reshape(params((1 + (n_hidden * (n_input + 1))):end), ...
                 n_class, (n_hidden + 1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(training_data, 1);

training_data = horzcat(training_data, ones(N, 1));

A = training_data * w1';

Z = sigmoid(A);
rows = size(Z, 1);
Z = horzcat(Z, ones(rows, 1));

B = Z * w2';

prediction = sigmoid(B);

training_label = oneofkencode(training_label, n_class);

% obj_val
obj_err = (sum(sum(training_label .* mylog(prediction))) + sum(sum((1 - training_label) .* mylog(1 - prediction)))) .* (-1) ./ N;
obj_reg = (sum(sum(w1 .* w1)) + sum(sum(w2 .* w2))) .* (lambda) ./ (2 * N);
obj_val = obj_err + obj_reg;

fprintf('\nobj_val is: %f\n', obj_val);
grad_w2 = ((prediction - training_label)' * Z) ./ N;
grad_w1 = (((1 - Z) .* Z) .* ((prediction - training_label) * w2))' * training_data;
grad_w1 = grad_w1(1:end-1,:);

% Suppose the gradient of w1 and w2 are stored in 2 matrices grad_w1 and grad_w2
% Unroll gradients to single column vector
obj_grad = [grad_w1(:) ; grad_w2(:)];

end
