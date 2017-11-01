function [mu, var, time] = gprMKA(K, Ks, Kss, y, hyp)

[n, ~] = size(K); [ns, ~] = size(Kss); na=n+ns;
sn2 = exp(2*hyp.lik);
params.dcore=ceil(hyp.k*(1+ns/n));
params.verbosity =0;
params.maxclustersize = 200;

G=[K Ks'; Ks Kss];
G=G+sn2*eye(na);

% construct an MMF of A
tic
mmf1 = MMF(G,params);
% C++ codes
% det = mmf1.determinant();
Kstar=mmf1.submatrix(1:n, n+1:na);
mmf1.invert();
U = mmf1.submatrix(1:na, n+1:na);
B = U(1:n,:);
D = U(n+1:na,:);
invD=pinv(D);
ypad = padarray(y,ns,'post');
Ay = mmf1.hit(ypad);
alpha = Ay(1:n,:)-B*invD*B'*y;
mu = Kstar'*alpha; 
time = toc;

invK=zeros(na);I=eye(na);
for j=1:n
    v = I(:,j);
    invK(:,j) = mmf1.hit(v);
end

var = diag(Kss-Kstar'*invK(1:n,1:n)*Kstar);

% nlZ = y'*alpha/2 + log(det)/2 + n*log(2*pi)/2;   
end