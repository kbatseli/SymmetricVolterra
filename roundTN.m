function a=roundTN(a,tol)
% a=roundTN(a,tol)
% ----------------
% Returns an approximation of the Tensor Network a such that the 
% approximation has a relative error tol.
%
% a         =   Tensor Network,
%
% tol 		=	scalar, relative approximation error.
%
% Reference
% ---------
%
% 2016, Kim Batselier

d=size(a.n,1);

% right to left orthogonalization via QR
for i=d:-1:2
   [Q,R]=qr(reshape(a.core{i},[a.n(i,1),prod(a.n(i,2:end))])',0);
   if size(Q,2) < a.n(i,1)
		a.n(i,end)=size(Q,1)/prod(a.n(i,2:end-1));
		a.n(i,1)=size(Q,2);
		a.core{i}=reshape(Q',[size(Q,2),prod(a.n(i,2:end-1)),size(Q,1)/prod(a.n(i,2:end-1))]);
        a.core{i-1}=reshape(reshape(a.core{i-1},[prod(a.n(i-1,1:end-1)),a.n(i-1,end)])*R',[a.n(i-1,1),prod(a.n(i-1,2:end-1)),size(R,1)]);
        a.n(i-1,end)=size(Q,2);
   else
		a.core{i}=reshape(Q(:,1:a.n(i,1))',[a.n(i,1),prod(a.n(i,2:end-1)),size(Q,1)/(prod(a.n(i,2:end-1)))]);
        a.core{i-1}=reshape(reshape(a.core{i-1},[prod(a.n(i-1,1:end-1)),a.n(i-1,end)])*R(1:a.n(i,1),:)',[a.n(i-1,1),prod(a.n(i-1,2:end-1)),a.n(i-1,end)]);
   end
end

% left to right compression via truncated SVD
delta=tol*norm(a.core{1}(:))/sqrt(d-1);
for i=1:d-1
    [U,S,V]=svd(reshape(a.core{i},[prod(a.n(i,1:end-1)),a.n(i,end)]),'econ');
    s=diag(S);
    r=0;
    while norm(s(end:-1:end-r)) < delta 
        r=r+1;
    end
    r=length(s)-r;
%     a.core{i}=reshape(U(:,1:r),[a.n(i,1),prod(a.n(i,2:end-1)),r]);
    a.core{i}=reshape(U(:,1:r),[a.n(i,1),a.n(i,2:end-1),r]);
    a.n(i,end)=r;
    a.core{i+1}=reshape((S(1:r,1:r)*V(:,1:r)')*reshape(a.core{i+1},[a.n(i+1,1),prod(a.n(i+1,2:end))]),[r,prod(a.n(i+1,2:end-1)),a.n(i+1,end)]);   
    a.n(i+1,1)=r;
end
 a.core{d}=reshape(a.core{d},a.n(d,:));

end
