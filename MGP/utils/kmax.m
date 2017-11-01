function [maxv,maxi]=kmax(v,k)
n=length(v);
maxi=1:k;
maxv=v(1:k);
[worstv,worsti]=min(maxv);
for i=k+1:n
    if(v(i)>=worstv)
        maxi(worsti)=i;
        maxv(worsti)=v(i);
        [worstv,worsti]=min(maxv);        
    end
end
end