function label = nnPredict(w1, w2, data)
% nnPredict predicts the label of data given the parameter w1, w2 of Neural
% Network.

% Input:
% w1: matrix of weights of connections from input layer to hidden layers.
%     w1(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% w2: matrix of weights of connections from hidden layer to output layers.
%     w2(i, j) represents the weight of connection from unit j in input 
%     layer to unit j in hidden layer.
% data: matrix of data. Each row of this matrix represents the feature 
%       vector of a particular image
       
% Output: 
% label: a column vector of predicted labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[rows, columns] = size(data);
data = horzcat(data, ones(rows, 1));
A = data * w1';
Z = sigmoid(A);
[rows, columns] = size(Z);
Z = horzcat(Z, ones(rows, 1));
B = Z * w2';
Y = sigmoid(B);

label = [];
[rows, columns] = size(Y);
for i = 1:1:rows
    [maxval, maxindex] = max(Y(i,:),[],2);
    label = vertcat(label, [zeros(1,(maxindex - 1)) ones(1, 1) zeros(1, 9 - (maxindex - 1))]);
end

end
