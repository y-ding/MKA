function K=computerKernel(X,Y,gamma)
K = exp(-sqdist(X,Y)/gamma); 
end