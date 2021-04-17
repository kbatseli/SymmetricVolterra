clear all, close all
clc
NOISE_ON = 0;

%% System generation
rng(1)
% rng('default')
d=7;
p=1;
l=1;
M=3;
n=p*M+1;
Ntrain=1000;
Nval=300;

r=100;
h=zeros(n^d,1);
% A=randn(n,r);
H=zeros(n,r);
for i=1:r,
    H(:,i)=abs(1e0*randn(1))*exp(-randi(10,1).*[1:n]);
end
u=randn(Ntrain+Nval,p);
U=makeU(u,M,1);
y=sum((U*H).^d,2);


%% rank vector construction
if mod(d,2)==1
    % odd degree
    ranks=zeros(1,(d-1)/2);
    for i=1:length(ranks)
        ranks(i)=nchoosek(n+i-1,n-1);
    end
    ranks=[ranks,fliplr(ranks)];
else
    %even degree
     ranks=zeros(1,d/2);
    for i=1:length(ranks)
        ranks(i)=nchoosek(n+i-1,n-1);
    end
    ranks=[ranks,fliplr(ranks(1:d/2-1))];
end

%% noise generation 
SNR=60;
if NOISE_ON
    norme=norm(y(1:Ntrain))/10^(SNR/20);
else
    norme=0;
end
e=randn(Ntrain,1);
e=e/norm(e)*norme;

%% identification via pseudo inverse
TN=rkh2tn(U(1:Ntrain,:)',d);
[Q,S,V]=svd(reshape(TN.core{d},[TN.n(d,1)*n,Ntrain]),'econ');
PE_rank=nchoosek(n+d-1,n-1);
Q=Q(:,1:PE_rank);S=S(1:PE_rank,1:PE_rank);V=V(:,1:PE_rank);
for i=1:d-1
   TN2.core{i}= TN.core{i};
   TN2.n(i,:)=TN.n(i,:);
end
TN2.core{d}=reshape((y(1:Ntrain)+e)'*V*inv(S)*Q',[TN.n(d,1),n,1,1]);
TN2.n(d,:)=[TN.n(d,1),n,1,1];
TN3=roundTN(TN2,1e-9);TN3.n  % uniform rank-3 Volterra kernels
h_hat3 = contract(TN3);
issym = norm(symcheck(reshape(h_hat3,n*ones(1,d))));
strcat("Symmetry coefficient of solution obtained via pseudoinverse: ",num2str(issym))
yhat=sim_volterraTN(u(Ntrain+1:end),transposeTN(TN3));
validation_error=norm(yhat(M+1:end)-y(Ntrain+M+1:end))/norm(y(Ntrain+1:end));

%% check symmetry per iteration with random initialization
MAXITR=3e1;
v=tic;
[TN,etrain,issym]=mvals(y(1:Ntrain)+e,u(1:Ntrain),M,ranks,1e-16,MAXITR);
toc(v)
subplot(2,1,1)
semilogy(etrain,'-d')
grid on
xlabel('Iteration')
ylabel('Training error')
subplot(2,1,2)
semilogy(issym,'-o')
grid on
xlabel('Iteration')
ylabel('Symmetry coefficient')
