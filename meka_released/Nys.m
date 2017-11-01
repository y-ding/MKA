
function [U] = Nys(data, m, k,gamma)
% standard Nystrom
% free to choose different approximations instead of Nystrom
[n,dim] = size(data);
dex = randperm(n);
center = data(dex(1:m),:);
W = exp(-sqdist(center, center)*gamma);
E = exp(-sqdist(data, center)*gamma);
[Ue, Va, Ve] = svd(full(W));
vak = diag(Va(1:k,1:k));
%pidx = find(va > 1e-10);
inVa = sparse(diag(vak.^(-0.5)));
U = E * Ue(:,1:k) * inVa;
