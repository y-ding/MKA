function [mu, var, time] = gprExact(K, Ks, Kss, y, hyp)
tic
[n,~]=size(K); [ns, ~] = size(Kss); 

sn2 = exp(2*hyp.lik);
L = chol(K/sn2+eye(n));               % Cholesky factor of covariance with noise
clear K %KRZ

alpha = solve_chol(L,y)/sn2;

% nlZ = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sn2)/2;  % -log marg lik
mu = Ks * alpha;  % predicted means

sW = ones(n,1)/sqrt(sn2);
V  = L'\(repmat(sW,1,ns).*Ks');
var = diag(Kss) - sum(V.*V,1)';

time = toc;

end