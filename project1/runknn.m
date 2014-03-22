function runknn(kvalue)


load('dataset.mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% **************K-Nearest Neighbors***************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = kvalue;
%   Test KNN with validation data
predicted_label = knnPredict(k, train_data, train_label, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
         mean(double(predicted_label == validation_label)) * 100);

 %  Test KNN with test data
predicted_label = knnPredict(k, train_data, train_label, test_data);
fprintf('\nTesting Set Accuracy: %f\n', ...
         mean(double(predicted_label == test_label)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *******Save the learned parameters *************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%save('params.mat', 'n_input', 'n_hidden', 'w1', 'w2', 'lambda', 'k');

exit();
end
