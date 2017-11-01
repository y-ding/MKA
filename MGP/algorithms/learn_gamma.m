function[hyp] = learn_gamma(method, trainX, trainY, hyp)

kfold = 3;
perc_train = 0.8;
gamma_vec = 0.125:0.125:1.5;
% gamma_vec = 0.25:0.25:1.5;
gamma_vec = gamma_vec.^2;
smse_vec = zeros(size(gamma_vec));
Best_smse = 100;

for j = 1:length(gamma_vec)
    gamma = gamma_vec(j);
    K = exp(-sqdist(trainX,trainX)/gamma);
    for k = 1:kfold
        ind = randperm(size(trainX,1));
        train = ind(1:round(perc_train*size(trainX,1)));
        test = ind((round(perc_train*size(trainX,1))+1):end);
        
        varTest=var(trainY(test));
        meanTest=mean(trainY(test));
        
        %========= Kernel matrix          
        
        if strcmp(method,'gprMeka')
            [mu, ~, ~] = feval('gprMeka',trainX(train),K(test,train), K(test,test), trainY(train),hyp);
        else
            [mu, ~, ~] = feval(method, K(train,train), K(test,train),K(test,test), trainY(train), hyp);
        end
        smse_vec(k) = mse(mu,trainY(test), meanTest, varTest);
        
    end
    clear K;
    mean_smse = mean(smse_vec);
    if mean_smse<Best_smse
        Best_smse = mean_smse;
        hyp.ell = gamma_vec(j);
    end
    
end