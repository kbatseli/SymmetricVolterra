%% System generation
clear all, close all
clc

rng(1) % for reproducibility
% rng('default')
D=10;
P=1;
L=1;
Nruns=10;
M=5;

oldtime = zeros(D,Nruns);
newtime = zeros(D,Nruns);

I=P*M+1;
R1 = nchoosek(D-1+I-1,I-1);
Ntrain=nchoosek(D+I-1,I-1);

U=makeU(randn(Ntrain,P),M,1);

%% experiment increasing D, fixed M
for i=1:Nruns
    for d=1:D,
        
        v=tic;
        TNref=rkh2tn(U',d);
        [Q,S,V]=svd(reshape(TNref.core{d},[TNref.n(d,1)*I,Ntrain]),'econ');
        PE_rank=nchoosek(d+I-1,I-1);
        Q=Q(:,1:PE_rank);S=S(1:PE_rank,1:PE_rank);V=V(:,1:PE_rank);    
        oldtime(d,i)=toc(v);

        if d==1
            v=tic;
            [VTN,St,Qt]=thinSVDTN(U,1);
            newtime(d,i)=toc(v);  
        else
            v=tic;
            [Vt,St,Qt]=svd(reshape(dotkron(Qt*St,U)',[nchoosek(d-1+I-1,I-1)*I,Ntrain]),'econ');
            VTN.n(d,1:4) = [nchoosek(d-1+I-1,I-1),I,1,nchoosek(d+I-1,I-1)];
            VTN.core{d} = reshape(Vt(:,1:nchoosek(d+I-1,I-1)),VTN.n(d,:));
            St=St(1:nchoosek(d+I-1,I-1),1:nchoosek(d+I-1,I-1));
            Qt=Qt(:,1:nchoosek(d+I-1,I-1));
            newtime(d,i)=toc(v);  
        end       
    end
end

avgtime=[sum(oldtime,2)/Nruns,sum(newtime,2)/Nruns,avgtime(:,1)./avgtime(:,2)]
semilogy(avgtime);hold on,semilogy(avgtime(:,1)./avgtime(:,2))