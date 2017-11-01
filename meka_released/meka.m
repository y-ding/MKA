function [U,S] = meka(A,k,gamma,opts)
% [U,S] = meka(A, k, gamma,opts)
% K \approx U*S*U^T 
%
% Arguments:
% A		       input data, an n by d sparse matrix, each row is a data point. 
% gamma        the kernel width in RBF kernel
% opts.noc	   number of clusters(default 10) 
% opts.eta     the percentage of blocks to be set to be 0(default 0.1)
%

% addpath('../../GPRApprox_Comparison/code/gpml')

if isempty(opts.noc)
	opts.noc = 10;
end
if isempty(opts.eta)
	opts.eta = 0.01;
end
noc = opts.noc;
eta = opts.eta;
[n,dim] = size(A);
% gamma = exp(hyp.cov(1));
% fprintf('***************************\n');
% fprintf('Parameters ...\n');
% fprintf('Rank = %d     gamma= %f \n',k, gamma);
% fprintf('# of clusters = %d eta= %f \n',opts.noc, opts.eta);
% fprintf('***************************\n');
% display('Begin meka!');
%
t = cputime;
%=============== k-means with sampling

l = 1:n;
MaxIter = 10; % number of iterations in kmeans
[idx1,centers] = mykmeans(A(l,:)',noc,MaxIter);
dis = sqdist(A,centers');
[v,idx] = min(dis');
D = exp(-sqdist(centers', centers')*gamma);

[sortedValues,sortIndex] = sort(D(:),'ascend');
mm = ceil((noc*noc-noc)*eta);
if eta==0,
	threshold = 0;
else
	threshold = sortedValues(mm);
end
prtc = cell(noc,1);
noc = numel(prtc);
for i = 1:noc,
	prtc{i} = find(idx==i);
	prtsz(i) = numel(prtc{i});
end

for i = 1:noc,
	klist(i) = ceil(k*prtsz(i)/sum(prtsz));
end
tt = sum(klist)-k;
if tt,
	klist(end) = klist(end)-tt;
end
% fprintf('The time for kmeans %f\n',cputime -t);
t = cputime;
%
%================= approximating diagonal blocks
indall = [];
for i = 1:noc,
	ind = prtc{i};
	indall = [indall;ind'];
	ki = klist(i);
	ki = min(ki,numel(ind));
	S{i,i} = eye(ki);
	m = 2*ki;
	m = min(m,numel(ind));
	[U{i}]=Nys(A(ind,:),m,ki,gamma); % standard nystrom approximation
	klist(i) = ki;
end
% fprintf('The time for approximating diagonal blocks %f\n',cputime -t);
%
t = cputime;
%================== approximating off-diagonal blocks

for i = 1:noc,
	ii = prtc{i};
	ki = klist(i);
	for j = (i+1):noc,
		jj = prtc{j};
		kj = klist(j);
	
		if D(i,j)<threshold, 
% if dis between two clusters are too large then set that off-diagonal to be 0
			S{i,j} = zeros(ki,kj);
			S{j,i} = zeros(kj,ki);
		else
			lii = numel(ii);
			ljj = numel(jj);
			randi = randsample(1:lii,min(4*ki,lii));
			randj = randsample(1:ljj,min(4*kj,ljj));
			Ui = U{i}(randi,:);
			Uj = U{j}(randj,:);
			tmpAi = A(ii(randi),:);
			tmpAj = A(jj(randj),:);
			tmpK = exp(-sqdist(tmpAi,tmpAj)*gamma);
			S{i,j} = pinv(full(Ui'*Ui),1e-6)*Ui'*tmpK*Uj*pinv(full(Uj'*Uj),1e-6);
			S{j,i} = S{i,j}';
		end
	end
end
% fprintf('The time for approximating off-diagonal blocks is %f\n',cputime-t);
%
[~,ind] = sort(indall);
U = blkdiag_meka(U);
U = U(ind,:);
S = sparse(cell2mat(S));
