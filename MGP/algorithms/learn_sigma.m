function[hyp] = learn_sigma(method, trainX, K, trainY, hyp)

kfold = 5;
perc_train = 0.90;
sigvec = -4:1;
smse_vec = zeros(size(sigvec));
Best_smse = 100;

for j = 1:length(sigvec)
    hyp.lik = sigvec(j);
    for k = 1:kfold
        ind = randperm(size(K,1));
        train = ind(1:round(perc_train*size(K,1)));
        test = ind((round(perc_train*size(K,1))+1):end);
        
        varTest=var(trainY(test));
        meanTest=mean(trainY(test));
        
        if strcmp(method,'gprMeka')
            [mu, ~, ~] = feval('gprMeka',trainX(train),K(test,train), K(test,test), trainY(train),hyp);
        else
            [mu, ~, ~] = feval(method, K(train,train), K(test,train),K(test,test), trainY(train), hyp);
        end
        smse_vec(k) = mse(mu,trainY(test), meanTest, varTest);
    end
    mean_smse = mean(smse_vec);
    if mean_smse<Best_smse
        Best_smse = mean_smse;
        hyp.lik = sigvec(j);
    end
    
end