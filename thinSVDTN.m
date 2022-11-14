function [VTN,St,Qt]=thinSVDTN(U,D)

% critical assumption is that u is Persistently Exciting

% U is NxI
[N,I]=size(U);
U=U'; % I x N 

[Vt,St,Qt]=svd(U,'econ');          
VTN.n(1,1:4) = [1,I,1,I];
VTN.core{1} = reshape(Vt,VTN.n(1,:));

for d=2:D
    [Vt,St,Qt]=svd(reshape(dotkron(Qt*St,U')',[nchoosek(d-1+I-1,I-1)*I,N]),'econ');
    VTN.n(d,1:4) = [nchoosek(d-1+I-1,I-1),I,1,nchoosek(d+I-1,I-1)];
    VTN.core{d} = reshape(Vt(:,1:nchoosek(d+I-1,I-1)),VTN.n(d,:));
    St=St(1:nchoosek(d+I-1,I-1),1:nchoosek(d+I-1,I-1));
    Qt=Qt(:,1:nchoosek(d+I-1,I-1));
end
end