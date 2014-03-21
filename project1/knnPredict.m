function label = knnPredict(k, train_data, train_label, test_data)
[train_rows,train_columns] = size(train_data);
[train_label_rows,train_label_columns] = size(train_label);
[test_data_rows,test_data_columns] = size(test_data);


if (train_label_columns > 1)
    Group = zeros(train_label_rows,1);
    for i = 1:1:train_label_rows
        [maxValue, maxIndex] = max(train_label(i,:),[],2);
        Group(i,1) = maxIndex;
    end
else
    Group = train_label;
end

%Sample = test_data;
%Training = train_data;


%Class = knnclassify(Sample, Training, Group)
%Class = knnclassify(Sample, Training, Group, k)
%Class = knnclassify(Sample, Training, Group, k, distance)
%Class = knnclassify(Sample, Training, Group, k, distance, rule)

single_column_label = knnclassify(test_data,train_data,Group,k);

int_column_label = int32(single_column_label);

label = zeros(test_data_rows,train_label_columns);

for i = 1:1:test_data_rows
    label(i,int_column_label(i,1)) = 1;
end

end

