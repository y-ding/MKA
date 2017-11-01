clear all;
close all;
maxNumCompThreads(1);
load ijcnn.mat;% input data matrix A should be sparse matrix with size n by d

%==================== parameters
k = 1280;
%k = 500; % target rank
gamma = 1; % kernel width in RBF kernel
opts.eta = 0.01; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 10; % number of clusters(default 10)

%==================== obtain the approximation U and S(K \approx U*S*U^T)
t = cputime;
[U,S] = meka(A,k,gamma,opts); % main function
display('Done with meka!');
fprintf('The total time cost for meka is %f secs\n',cputime -t);
fprintf('***************************\n');
%==================== measure the relative error
display('Testing meka!');
[n,d] = size(A);
rsmp = 1000; % sample several rows in K to measure kernel approximation error
rsmpind = randsample(1:n,rsmp);
tmpK = exp(-sqdist(A(rsmpind,:),A)*gamma);
Err = norm(tmpK-(U(rsmpind',:)*S)*U','fro')/norm(tmpK,'fro')
fprintf('The relative approximation error is %f\n',Err);
