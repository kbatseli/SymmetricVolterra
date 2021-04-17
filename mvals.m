function [TN,e,varargout]=mvals(y,u,M,r,varargin)
% [TN,e]=mvals(y,u,M,r) or [TN,e]=mvals(y,u,M,r,THRESHOLD,MAXITR)
% ---------------------------------------------------------------
% MIMO Volterra Alternating Linear Scheme (MVALS) algorithm for 
% solving the MIMO Volterra system identification problem in the Tensor
% Network format.
%
% TN        =   Tensor Network, TN.core is a cell containing the TN-cores,
%               TN.n is a matrix where TN.n(i,:) are the dimensions of the
%               ith TN-core,
%
% e         =   vector, e(i) contains the relative residual 
% 				||y-yhat||_2/||y||_2 at iteration i,
%
% y         =   matrix, y(:,k) contains the kth output,
%
% u         =   matrix, u(:,k) contains the kth input,
%
% M         =   scalar, memory of each of the Volterra kernels,
%
% r         =   vector, contains the TT ranks r_1 up to r_{d-1}, since
%               r_0=r_d=1.
%
% THRESHOLD =   scalar, optional threshold on RMS error to stop iterations.
%               Default=1e-4.    
%
% MAXITR    =   scalar, optional maximum number of iterations. Default=100.    
%
% Reference
% ---------
%
% 06/07/11 - 2016, Kim Batselier

% rng('default');

p=size(u,2);                    % number of inputs
[N,l]=size(y);
y=reshape(y',[N*l,1]);
d=length(r)+1;                  % degree of truncated Volterra series
r=[l r(:)' 1];                  % append extremal TN ranks
n=p*M+1;
if ~isempty(varargin)
    THRESHOLD=varargin{1};
    MAXITR=varargin{2};
else
    THRESHOLD=1e-16;
    MAXITR=20;
end

% construct N x n matrix U
U=zeros(N,n);
u=[zeros(M-1,p);u];
for i=M:N+M-1            
	temp=ones(1,n);
    for j=1:M
        temp(2+(j-1)*p:2+j*p-1)=u(i-j+1,:);                
    end   
    U(i-M+1,:)=temp;
end
u=u(M:end,:);
Vp=cell(1,d);
Vm=cell(1,d);
if l==1
    Vm{1}=ones(N,1);
else
    Vm{1}=eye(l);
end
Vp{d}=ones(N,1);

% initialize right-orthonormal cores with prescribed TN ranks
TN.core=cell(1,d);
TN.core{1}=rand(r(1),n,r(2));
TN.core{1}=TN.core{1}./norm(TN.core{1}(:));
TN.n(1,:)=[1 l n r(2)];
for i=d:-1:2
	TN.n(i,:)=[r(i) 1 n r(i+1)];
    TN.core{i}=permute(reshape(orth(rand((n)*r(i+1),r(i))),[r(i+1),(n),r(i)]),[3,2,1]);    
    Vp{i-1}=dotkron(Vp{i},U)*reshape(permute(TN.core{i},[3 2 1]),[r(i+1)*n,r(i)]); % N x r_{i-1}    
end

e(1)=THRESHOLD+1;               % We always do at least 1 core update
itr=1;                          % counts number of iterations
ltr=1;                          % flag that checks whether we sweep left to right
sweepindex=1;                   % index that indicates which TT core will be updated
issym=zeros(1,MAXITR);

while (itr < MAXITR) && e(itr) > THRESHOLD
    updateTT;
    updatesweep;
end  

varargout{1}=issym;

    function updateTT
        % first construct the linear subsystem matrix
        if l==1
            A=dotkron(Vm{sweepindex},U,Vp{sweepindex});
        elseif sweepindex == 1
            A=kron(dotkron(U,Vp{sweepindex}),Vm{sweepindex});
        else
            A=dotkron(Vm{sweepindex},U,Vp{sweepindex});
            A=reshape(A,[N,l,r(sweepindex)*n*r(sweepindex+1)]);
            A=permute(A,[2 1 3]);
            A=reshape(A,[N*l,r(sweepindex)*n*r(sweepindex+1)]);
        end 
%         %% rank-dependent solver
%         s=svd(A);
%         tol = max(size(A)) * eps(max(s));
%         tempr = sum(s > tol);
%         if tempr<size(A,2)
%             g=pinv(A)*y;
%         else
%             g=A\y;
%         end
        
        %% pseudo-inverse
        g=pinv(A)*y;
%         g=(A)\y;
%         g=(A+10^3*eye(size(A)))\y;

		% contract the network to check symmetry
        hhat=reshape(TN.core{1},[prod(TN.n(1,2:3)),TN.n(1,4)]);
        for i=2:d
            hhat=hhat*reshape(TN.core{i},[TN.n(i,1),prod(TN.n(i,2:end))]);
            hhat=reshape(hhat,[n^i,TN.n(i,end)]);
        end
        issym(itr) = norm(symcheck(reshape(hhat,n*ones(1,d))));
        itr=itr+1;
        e(itr)=norm(A(l*M+1:end,:)*g-y(l*M+1:end))/norm(y(l*M+1:end));
%         
        
        if ltr
            % left-to-right sweep, generate left orthogonal cores and update vk1
            [Q,R]=qr(reshape(g,[r(sweepindex)*(n),r(sweepindex+1)])); 
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n,r(sweepindex+1)]);
            TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n)*r(sweepindex+2)]),[r(sweepindex+1),n,r(sweepindex+2)]);
            if l==1
                Vm{sweepindex+1}=dotkron(Vm{sweepindex},U)*reshape(TN.core{sweepindex},[r(sweepindex)*n,r(sweepindex+1)]); % N x r_{i}
            elseif sweepindex==1
                Vm{sweepindex+1}=U*reshape(permute(TN.core{sweepindex},[2 1 3]),[n,r(sweepindex)*r(sweepindex+1)]); %N x r_{i-1}r_i
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,r(sweepindex)*r(sweepindex+1)]);                
            else
                Vm{sweepindex+1}=reshape(dotkron(Vm{sweepindex},U),[N*l,r(sweepindex)*n])*reshape(TN.core{sweepindex},[r(sweepindex)*n,r(sweepindex+1)]);
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,l*r(sweepindex+1)]);
            end
        else
            % right-to-left sweep, generate right orthogonal cores and update vk2
            [Q,R]=qr(reshape(g,[r(sweepindex),(n)*r(sweepindex+1)])'); 
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex))',[r(sweepindex),n,r(sweepindex+1)]);
            TN.core{sweepindex-1}=reshape(reshape(TN.core{sweepindex-1},[r(sweepindex-1)*(n),r(sweepindex)])*R(1:r(sweepindex),:)',[r(sweepindex-1),n,r(sweepindex)]);
            Vp{sweepindex-1}=dotkron(Vp{sweepindex},U)*reshape(permute(TN.core{sweepindex},[3 2 1]),[r(sweepindex+1)*n,r(sweepindex)]); % N x r_{i-1}    

        end
    end


    function updatesweep
        if ltr
            sweepindex=sweepindex+1;
            if sweepindex== d                
                ltr=0;
            end
        else
            sweepindex=sweepindex-1;
            if sweepindex== 1                
                ltr=1;
            end
        end
    end
end
