function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

k = size(a,1);
l = size(b,1);
d = sum(a.*a,2)*ones(1,l)+ones(k,1)*(sum(b.*b,2))'-2*a*b';

