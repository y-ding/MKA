clc
clear all;
%==================== add library
addpath('../../../pMMF/matlab')
addpath('../../meka_released')
addpath('../algorithms')
addpath('../cov')
addpath('../utils')

%==================== load data
dataset = 'rupture'; %k=64 0.25:0.25:1.5;
load(sprintf('../data/%s',dataset));

%==================== parameters
[n,D]= size(X);
ratio = 0.9;
eta = (1-ratio)/ratio+1;
n_train = floor(n*ratio);       % size of subset of the data for each resampling run 
n_test = n-n_train;
sqrt_nstar = sqrt(n_test);

mrank = 4:8;
m_rank = 2.^mrank;
nsamples = 1;     % number of samples in resampling
sigma = 0;
hyp.lik = sigma; hyp.ell=1;

perm = randperm(n);
INDEX_s = perm(1:n_train);INDEX_t = perm(n_train+1:n_train+n_test);
trainX = X(INDEX_s,:);
trainY = y(INDEX_s,:);    
testX=X(INDEX_t,:);
testY=y(INDEX_t,:);

%------------------------------------------------------------------------
% Normalize data to zero mean, variance one. 
%------------------------------------------------------------------------
meanMatrix = repmat(mean(trainX), n_train, 1);
trainYMean = mean(trainY);
trainYStd  = std(trainY);
stdMatrix  = repmat(std(trainX), n_train, 1);
trainX = (trainX - meanMatrix);
trainX = trainX./stdMatrix;
trainY = (trainY - trainYMean)./trainYStd;

testX  = (testX-repmat(meanMatrix(1,:), size(testX,1),1))./repmat(stdMatrix(1,:), size(testX,1),1);
testY  = (testY - trainYMean)./trainYStd;

varTest=var(testY);
meanTest=mean(testY);

%==================== parameters initialize

t_v = zeros(nsamples,numel(m_rank));
t_s = zeros(nsamples,numel(m_rank));
t_f = zeros(nsamples,numel(m_rank));
t_p = zeros(nsamples,numel(m_rank));
t_m = zeros(nsamples,numel(m_rank));
t_meka = zeros(nsamples,numel(m_rank));

smse_v = zeros(nsamples,numel(m_rank));
smse_s = zeros(nsamples,numel(m_rank));
smse_f = zeros(nsamples,numel(m_rank));
smse_p = zeros(nsamples,numel(m_rank));
smse_m = zeros(nsamples,numel(m_rank));
smse_meka = zeros(nsamples,numel(m_rank));

mnlp_v = zeros(nsamples,numel(m_rank));
mnlp_s = zeros(nsamples,numel(m_rank));
mnlp_f = zeros(nsamples,numel(m_rank));
mnlp_p = zeros(nsamples,numel(m_rank));
mnlp_m = zeros(nsamples,numel(m_rank));
mnlp_meka = zeros(nsamples,numel(m_rank));

for i = 1:nsamples 
    for j=1:numel(m_rank) 
        
    hyp.k = m_rank(j); % number of pseudo inputs

    % gprExact
    fun = 'gprExact';
    [hyp] = learn_gamma(fun, trainX, trainY, hyp); 
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    [mu_v, var_v, time_v] = gprExact(K, Ks, Kss, trainY, hyp); 
    t_v(i,j) = time_v;
    smse_v(i,j) = mse(mu_v,testY, meanTest, varTest);
    mnlp_v(i,j) = mnlp(mu_v,testY, var_v, meanTest, varTest);
    fprintf('Exact: smse =  %f, mnlp= %f, time = %f \n',smse_v(i,j),mnlp_v(i,j), t_v(i,j));    
    
    % gprSOR 
    fun = 'gprSOR';
    [hyp] = learn_gamma(fun, trainX, trainY, hyp); 
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    [mu_s, var_s, time_s] = gprSOR(K, Ks, Kss, trainY, hyp);
    t_s(i,j)=time_s;
    smse_s(i,j) = mse(mu_s,testY, meanTest, varTest);
    mnlp_s(i,j) = mnlp(mu_s,testY, var_s, meanTest, varTest);
    fprintf('SOR: m= %d, smse =  %f, mnlp= %f, time = %f \n',hyp.k, smse_s(i,j),mnlp_s(i,j), t_s(i,j));    
   
    % gprFITC 
    fun = 'gprFITC';
    [hyp] = learn_gamma(fun, trainX, trainY, hyp); 
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    [mu_f, var_f, time_f] = gprFITC(K, Ks, Kss, trainY, hyp);
    t_f(i,j)=time_f;
    smse_f(i,j) = mse(mu_f,testY, meanTest, varTest);
    mnlp_f(i,j) = mnlp(mu_f,testY, var_f, meanTest, varTest);
    fprintf('FITC: m= %d, smse =  %f,  mnlp= %f, time = %f \n',hyp.k, smse_f(i,j),mnlp_f(i,j), t_f(i,j));

     % gprPITC 
    fun = 'gprPITC';
    [hyp] = learn_gamma(fun, trainX, trainY, hyp); 
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    [mu_p, var_p, time_f] = gprPITC(K, Ks, Kss, trainY, hyp);
    t_p(i,j)=toc;
    smse_p(i,j) = mse(mu_p,testY, meanTest, varTest);
    mnlp_p(i,j) = mnlp(mu_p,testY, var_p, meanTest, varTest);
    fprintf('PITC: m= %d, smse =  %f,  mnlp= %f, time = %f \n',hyp.k, smse_p(i,j),mnlp_p(i,j), t_p(i,j));
 
    % gprMeka
    fun = 'gprMeka';
    %%%%%%[hyp] = learn_gamma(fun, trainX, trainY, hyp); 
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    tic
    [mu_meka, var_meka] = gprMeka(trainX, Ks, Kss, trainY, hyp);
    t_meka(i,j)=toc;
    smse_meka(i,j) = mse(mu_meka,testY, meanTest, varTest);
    mnlp_meka(i,j) = mnlp(mu_meka,testY, var_meka, meanTest, varTest);
    fprintf('Meka: m= %d, smse =  %f, mnlp= %f, time = %f \n',hyp.k, smse_meka(i,j), mnlp_meka(i,j), t_meka(i,j));  
   
    % gprMKA
    fun = 'gprMKA';
    [hyp] = learn_gamma(fun, trainX, trainY, hyp);
    gamma = hyp.ell^2;
    K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
    [mu_m, var_m, time_m] = gprMKA(K, Ks, Kss, trainY, hyp);
    t_m(i,j)=time_m;
    smse_m(i,j) = mse(mu_m,testY, meanTest, varTest);
    mnlp_m(i,j) = mnlp(mu_m,testY, var_m, meanTest, varTest);
    fprintf('MKA: m= %d, smse =  %f, mnlp= %f, time = %f \n',hyp.k, smse_m(i,j), mnlp_m(i,j), t_m(i,j));  
    
    end
end     

meanV = mean(smse_v,1);
meanS = mean(smse_s,1);
meanF = mean(smse_f,1);
meanP = mean(smse_p,1);
meanMeka = mean(smse_meka,1);
meanM = mean(smse_m,1);

meanPV = mean(mnlp_v,1);
meanPS = mean(mnlp_s,1);
meanPF = mean(mnlp_f,1);
meanPP = mean(mnlp_p,1);
meanPMeka = mean(mnlp_meka,1);
meanPM = mean(mnlp_m,1);

meanTV = mean(t_v,1);
meanTS = mean(t_s,1);
meanTF = mean(t_f,1);
meanTP = mean(t_p,1);
meanTMeka = mean(t_meka,1);
meanTM = mean(t_m,1);


save(sprintf('result/%s_rank',dataset),'meanV','meanS','meanF','meanP','meanMeka','meanM',...
    'meanPV','meanPS','meanPF','meanPP','meanPMeka','meanPM',...
    'meanTV','meanTS','meanTF','meanTP','meanTMeka','meanTM');

METHODS = {'Exact', 'SOR','FITC', 'PITC','MEKA','MKA'}; % Plot data for these methods only.
plot_colors = {'k-d', 'c-d', 'b-d', 'g-d', 'm-d', 'r-d'}; % At least as many colors as methods 

gray = [112 118 115]./255;

figure
lw =4;
fs = 24;
plot(log2(m_rank), meanV,'k--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanS,'c--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanF,'b--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanP,'g--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanMeka,'m--d', 'Linewidth', lw);
hold on
plot(log2(m_rank), meanM,'r--d', 'Linewidth', lw);
axis tight;
xlabel('Log_2 # pseudo-inputs','FontSize',fs,'FontWeight','bold');
ylabel('SMSE','FontSize',fs,'FontWeight','bold');
lngd = legend('Full','SOR','FITC','PITC','MEKA','MKA','Location','NorthEast');
set(lngd, 'fontsize', 14, 'Position', [0.55,0.55,0.15,0.25]);

figure
lw =4;
plot(log2(m_rank), meanPV,'k--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanPS,'c--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanPF,'b--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanPP,'g--d','Linewidth', lw);
hold on
plot(log2(m_rank), meanPMeka,'m--d', 'Linewidth', lw);
hold on
plot(log2(m_rank), meanPM,'r--d', 'Linewidth', lw);
axis tight;
xlabel('Log_2 # pseudo-inputs','FontSize',fs,'FontWeight','bold');
ylabel('MNLP','FontSize',fs,'FontWeight','bold');
lngd = legend('Full','SOR','FITC','PITC','MEKA','MKA','Location','NorthEast');
set(lngd, 'fontsize', 14, 'Position', [0.55,0.55,0.15,0.25]);

