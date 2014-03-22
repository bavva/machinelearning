function oneofk = oneofkencode(label, k)
% ONEOFKENCODE takes input a label and returns 
% one-of-k type matrix
% note: label is 0 to k - 1

rows = size(label, 1);
oneofk = zeros(rows, k);

for i = 1:rows
    column = label(i, 1) + 1;
    oneofk(i, column) = 1;
end

end