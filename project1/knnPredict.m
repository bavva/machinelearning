function label = knnPredict(k, train_data, train_label, test_data)
[train_rows,train_columns] = size(train_data);
[train_label_rows,train_label_columns] = size(train_label);
[test_rows,test_columns] = size(test_data);


% 
%   IF train_label IS NOT A SINGLE COLUMN MATRIX/VECTOR, USE THE BELOW CODE TO CONVERT IT
%
%if (train_label_columns > 1)
%    Group = zeros(train_label_rows,1);
%    for i = 1:1:train_label_rows
%        [maxValue, maxIndex] = max(train_label(i,:),[],2);
%        Group(i,1) = maxIndex;
%    end
%else

%end

%Sample = test_data;
%Training = train_data;
%Group = train_label;

%Class = knnclassify(Sample, Training, Group)
%Class = knnclassify(Sample, Training, Group, k)
%Class = knnclassify(Sample, Training, Group, k, distance)
%Class = knnclassify(Sample, Training, Group, k, distance, rule)
label = zeros(test_rows,1);            
if (train_label_columns == 1)
    if (train_rows == train_label_rows)
        if (train_columns == test_columns)
%       COMMENT THE BELOW LINE TO RUN WITHOUT USING THE DEFAULT KNNCLASSIFY FUNCTION
            label = knnclassify(test_data,train_data,train_label,k);
            
%
%           CODE WITHOUT USING KNNCLASSIFT() STARTS HERE
%
%             [gIndex,groups] = grp2idx(train_label);
%             for i = 1:1:test_rows
%                 column_dists_each_test = zeros(train_rows,1);
%                 for j = 1:1:train_rows
%                     dist = sum((train_data(j,:) - test_data(i,:)).^2,2);
%                     %column_dists_each_test = sum((train_data - test_data(repmat(i,train_rows,1),:)).^2, 2);
%                     column_dists_each_test(j,1) = dist;
%                 end
%                 [dSorted,dIndex] = sort(column_dists_each_test);
%                 dIndex = dIndex(1:k,1);
%                 possible_classes = gIndex(dIndex);
%                 label(i,1) = mode(possible_classes);
%             end
%             if isnumeric(train_label) || islogical(train_label)
%                 groups = str2num(char(groups)); %#ok
%                 label = groups(label);
%             elseif ischar(train_label)
%                 groups = char(groups);
%                 label = groups(label,:);
%             else %if iscellstr(group)
%                 label = groups(label);
%             end            
%            
%           CODE WITHOUT USING KNNCLASSIFY() ENDS HERE
%
        else
            sprintf('Training data and Test data feature count not matching!!');
        end
    else
        sprintf('Training data and Training label have different number of rows!!');
    end
else
    sprintf('Training label should be a single column matrix!!');
end

end

