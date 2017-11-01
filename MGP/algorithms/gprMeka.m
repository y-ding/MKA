function [mu, var, time] = gprMeka(X, Ks, Kss, y, hyp)

opts.eta = 0.01; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 5; % number of clusters(default 10)
k = hyp.k;
%==================== obtain the approximation U and S(K \approx U*S*U^T)
[n,~]=size(X);[ns, ~] = size(Kss); 
X=sparse(X);
gamma = 1/hyp.ell;
tic
[U,S] = meka(X,k,gamma,opts); % main function
K = U*S*U'; 
reg = 1e1*12;
sn2  = exp(2*hyp.lik); 
try
    L = chol(K/sn2+reg*eye(n));               % Cholesky factor of covariance with noise
    clear K %KRZ    
    alpha = solve_chol(L,y)/sn2;      
    
    mu = Ks * alpha;  % predicted means    
    sW = ones(n,1)/sqrt(sn2);
    V  = L'\(repmat(sW,1,ns).*Ks');
    var = diag(Kss) - sum(V.*V,1)';
       
%     nlZ = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sn2)/2;  % -log marg lik

catch
    % reg = 10*reg;
    % L = chol(K/sn2+reg*eye(n));
    mu = NaN;
    var = NaN;
end
time = toc;

end
