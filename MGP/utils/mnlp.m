function err = mnlp(estimatedYs, realYs, vars, meanTest, varTest)
% Return the normalized MNLP (or MSLL). 
% NOTE: This assumes the training outputs are mean 0 var 1!
%
% Krzysztof Chalupka, University of Edinburgh 2011
err = 0.5*mean(((realYs - estimatedYs)./sqrt(vars)).^2 + log(vars)) - 0.5*(varTest + meanTest^2);
