function label = oneofkdecode(oneofk)
% ONEOFKDECODE takes input a oneofk matrix
% and returns a label
% note: label is 0 to k - 1

[C, label] = max(oneofk, [], 2);
label = label - 1;

end