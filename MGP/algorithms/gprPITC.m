function[mu, var, time] = gprPITC(K, Ks, Kss, y, hyp)
tic
[n, ~] = size(K);
sigma  = exp(2*hyp.lik); 

% choose a random set of m_rank indices for the active set
perm = randperm(n);
u = perm(1:hyp.k);
m=length(u);
tic;

[L, L_partition] = getLambda(K,getQ(K,1:size(K,1),1:size(K,1),u),sigma,m);
Sigma = getS(K,L,u,L_partition);
% R = chol(Sigma); invSigma = solve_chol(R,ones(size(R,1),1));
mu = Ks(:,u)*(pinv(Sigma)*(K(u,:)*solveLambda(L,L_partition,y)));

Qstar = Ks(:,u)*(pinv(K(u,u))*(Ks(:,u)'));
var = diag(Kss - Qstar + Ks(:,u)*(pinv(Sigma)*Ks(:,u)'));
time = toc;
end

function[Q] = getQ(K,ainds,binds,uinds)
Q = K(ainds,uinds)*(pinv(K(uinds,uinds))*K(uinds,binds));
end

function[SigmaI] = getS(K, Lambda, uinds,partition)
SigmaI = K(uinds,uinds) + K(uinds,:)*solveLambda(Lambda,partition,K(:,uinds));
end

function[v] = solveLambda(Lambda,partition,y)
v = zeros(size(Lambda,1),size(y,2));
for curb = 1:size(partition,1)
    inds = partition(curb,1):partition(curb,2);
    v(inds,:) = pinv(Lambda(inds,inds))*y(inds,:);
end
end


function[L,L_partition] = getLambda(K,Q,sigma,m)

num_blocks = floor(size(K,1)/m);
L_partition = zeros(num_blocks,2);
L_partition(:,1) = (1:num_blocks)*m - m + 1;
L_partition(:,2) = L_partition(:,1) + m -1;
L_partition(end,2) = size(K,1);

L_pre = K - Q + sigma*eye(size(K));
L = zeros(size(K));
for curb = 1:num_blocks
    inds = L_partition(curb,1):L_partition(curb,2);
    L(inds,inds) = L_pre(inds,inds);
end

end