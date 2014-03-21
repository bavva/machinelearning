function g = mylog(z)
% mylog computes log function and returns 0 if z = 0

g = [];

[rows, columns] = size(z);
for i = 1:rows
    for j = 1:columns
        if (z(i, j) > 0)
            val = log(z(i, j));
        else
            val = 0;
        end
        g(i, j) = val;
    end
end

end
