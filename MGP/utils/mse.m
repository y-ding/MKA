function msError = mse(estimatedYs, realYs, meanTest, varTest)
% Return the normalized MSE (SMSE).
% NOTE: This assumes the training outputs are mean 0 var 1!
%
% Krzysztof Chalupka, University of Edinburgh 2011
msError = mean((estimatedYs-realYs).^2)/(varTest+meanTest^2);
