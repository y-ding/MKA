function[mu, var, time] = gprFITC(K, Ks, Kss, y, hyp)
tic
[n, ~] = size(K);[ns, ~] = size(Kss); 
I = ones(n,1);
nu = hyp.k;
perm = randperm(n);    % choose a random set of m_rank indices for the active set
idx = perm(1:nu);
Kuu = K(idx,idx);
Ku = K(idx,:);

sn2  = exp(2*hyp.lik);                              % noise variance of likGauss
snu2 = 1e-6*sn2;                              % hard coded inducing inputs noise
Luu  = chol(Kuu + snu2*eye(nu));                       % Kuu + snu2*I = Luu'*Luu
V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q
dg = diag(K) + sn2 - sum(V.*V,1)';
V  = V./repmat(sqrt(dg)',nu,1);
Lu = chol(eye(nu) + V*V');
r  = y./sqrt(dg);
be = Lu'\(V*r);
iKuu = solve_chol(Luu,eye(nu));

L  = solve_chol(Lu*Luu,eye(nu)) - iKuu; % Sigma-inv(Kuu)
alpha = Luu\(Lu\be);                      % return the posterior parameters

Ksu = Ks(:,idx);
mu = Ksu * alpha;  % predicted means
% nlZ = sum(log(diag(Lu))) + (sum(log(dg)) + n*log(2*pi) + r'*r - be'*be)/2; 
var = diag(Kss) + sum(Ksu'.*(L*Ksu'),1)';
time = toc;
end