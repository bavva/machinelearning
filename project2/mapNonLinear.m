function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))

x_original = x;
x_n = ones(size(x,1),d+1);
for i=1:1:d
    x_n(:,i+1) = x_original.^i;
end

end
