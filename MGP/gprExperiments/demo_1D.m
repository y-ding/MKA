clc
clear all;
%==================== add library
addpath('../../../pMMF/matlab')
addpath('../../meka_released')
addpath('../algorithms')
addpath('../cov')
addpath('../utils')

dataset = 'Snelson1D';
load(sprintf('../data/%s',dataset));

trainX=Xs; testX=Xs;X=Xs;

n=length(X);
gamma = 4;
K = exp(-sqdist(X,X)*gamma);

mu=zeros(1,n);
mv=mvnrnd(mu,eye(n),1);

[V,D] = eig(K);
S=sqrt(D);
ground_truth=V*D*mv';
y=ground_truth+randn(n,1);

trainY=y;

hyp.k = 10;
sigma = 0;
hyp.lik = sigma; hyp.ell=1;

% gprExact
fun = 'gprExact';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_v, var_v,~] = gprExact(K, K, K, trainY, hyp);

% gprSOR
fun = 'gprSOR';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_s, var_s,~] = gprSOR(K, K, K, trainY, hyp);

% gprFITC
fun = 'gprFITC';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_f, var_f,~] = gprFITC(K, K, K, trainY, hyp);

% gprPITC
fun = 'gprPITC';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_p, var_p] = gprPITC(K, K, K, trainY, hyp);

% gprMeka
fun = 'gprMeka';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_meka, var_meka,~] = gprMeka(trainX, K, K, trainY, hyp);

% gprMKA
fun = 'gprMKA';
[hyp] = learn_gamma(fun, trainX, trainY, hyp);
gamma = hyp.ell^2;
K = computerKernel(trainX,trainX,gamma); Ks = computerKernel(testX,trainX,gamma); Kss = computerKernel(testX,testX,gamma);
[mu_m, var_m,~] = gprMKA(K, K, K, trainY, hyp);

gray = [112 118 115]./255;
orange =[25 25 58] ./ 255;
m_rank =1:n;

lw = 4;
fs = 20;
idx = 1:6:301;
idxP = 1:3:301;

save('../toy/1D_toy','m_rank','idx','trainY','mu_v','mu_s','mu_f','mu_p','mu_meka','mu_m',...
    'var_v','var_s','var_f','var_p','var_meka','var_m');

%%=============== NIPS 2017 MODIFICATION

v = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_v(idx),'k','Linewidth', lw);
hold on
plot(m_rank(idx), mu_v(idx) + 2 * sqrt(var_v(idx)),'k--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_v(idx) - 2 * sqrt(var_v(idx)),'k--','Linewidth', lw);
axis tight;
saveas(v, '../toy/Full.pdf');

s = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_s(idx),'c','Linewidth', lw);
hold on
plot(m_rank(idx), mu_s(idx) + 2 * sqrt(var_s(idx)),'c--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_s(idx) - 2 * sqrt(var_s(idx)),'c--','Linewidth', lw);
axis tight;
saveas(s, '../toy/SOR.pdf');

f = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_f(idx),'b','Linewidth', lw);
hold on
plot(m_rank(idx), mu_f(idx) + 2 * sqrt(var_f(idx)),'b--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_f(idx) - 2 * sqrt(var_f(idx)),'b--','Linewidth', lw);
axis tight;
saveas(f, '../toy/FITC.pdf');

p = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_p(idx),'g','Linewidth', lw);
hold on
plot(m_rank(idx), mu_p(idx) + 2 * sqrt(var_p(idx)),'g--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_p(idx) - 2 * sqrt(var_p(idx)),'g--','Linewidth', lw);
axis tight;
saveas(p, '../toy/PITC.pdf');

meka = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_meka(idx),'m', 'Linewidth', lw);
hold on
plot(m_rank(idx), mu_meka(idx) + 2 * sqrt(var_meka(idx)),'m--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_meka(idx) - 2 * sqrt(var_meka(idx)),'m--','Linewidth', lw);
axis tight;
saveas(meka, '../toy/Meka.pdf');

m = figure;
plot(m_rank(idx), trainY(idx),'o','Color',orange,'Linewidth', 2);
hold on
plot(m_rank(idx), mu_m(idx),'r', 'Linewidth', lw);
hold on
plot(m_rank(idx), mu_m(idx) + 2 * sqrt(var_m(idx)),'r--','Linewidth', lw);
hold on
plot(m_rank(idx), mu_m(idx) - 2 * sqrt(var_m(idx)),'r--','Linewidth', lw);
axis tight;
saveas(m, '../toy/MKA.pdf');





