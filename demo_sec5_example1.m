% set number of Monte Carlo runs in line 65

% statistics for two sample test
% (1) gaussian mmd
% (2) akmmd-L2
% (3) akmmd-spec
% (4) KS-randproj


    % You are free to use, change, or redistribute this code in any way you
    % want for non-commercial purposes. However, it is appreciated if you 
    % maintain the name of the original author, and cite the paper:
    % X. Cheng, A. Cloninger, R. Coifman.  "Two Sample Statistics Based on Anisotropic Kernels."
    % arxiv:1709.05006
    %
    % Date: October 20, 2017. (Last Modified: October 20, 2017)

function demo_sec5_example1()

clear all;close all;
rng(20170807);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameter of data 

delta_max=.02;

epsx=.02;

dim=2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generation of referece set by heuristic
sig2g= epsx.^2;
sig2m =1;

nR=100;
[R,gmR]=generate_ref_local_pca(nR, @(n)generate_curve_data(n,delta_max,epsx), sig2g);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% two sample test

ntest=200; %n1,n2

nX=ntest;
nY=ntest;

%% parameter of kernel bandwidth
sig2_list= [1/4,1/2,1,2,4].^2;
nrow=numel(sig2_list);

%% deviation of q from p

del_list = (0:.2:1)*delta_max;

ncol=numel(del_list);

%%
lspec=20;

numspec=min(nR,ntest);

ll=(1:1:numspec)';
targetspec=exp(-2*(ll-(lspec-4)))./(exp(-2*(ll-(lspec-4)))+1);


%% under H0
alp=.05; %level of test

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set number of Monte Carlo runs

nrun = 100;

% nrun= 1000;
    % to reproduce figures in the paper, use nrun= 1000

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
numrp=20; %number of random projections in KS test

%

vote_g=zeros(nrow,ncol,nrun);
vote_m=zeros(nrow,ncol,nrun);
vote_mspec=zeros(nrow,ncol,nrun);
vote_ks=zeros(1,ncol,nrun);

numperm=40; %number of permutations to estimate threshold


%%


for icol=1:ncol
    
    delta=del_list(icol);
    fprintf('del=%6.4f:',delta)
    
    for irun=1:nrun
        
        if mod(irun,10)==0
            fprintf('-%d-',irun)
        end
        
        [X,Y]=generate_curve_data(ntest,delta,epsx);
        
        %% gmmd
        D2=euclidean_dis2(X,Y);
        for irow=1:nrow
            s2=sig2g*sig2_list(irow);
            K1=exp(-D2/(2*s2));
         
            eta1=calculate_kernel_mmd2(K1,nX,nY);
            
            etastore=zeros(1,numperm);
            for iboot=1:numperm
                idx=randperm(nX+nY);
                etastore(iboot)=calculate_kernel_mmd2(K1(idx,idx),nX,nY);
            end
            talp=quantile(etastore,1-alp);
            
            vote_g(irow,icol,irun)=double(eta1>talp);
        end
        
        %% akmmd
        [D2X,D2Y]=mahal_dis2(X,Y,gmR);
        for irow=1:nrow
            s2=sig2m*sig2_list(irow);
            A=[exp(-D2X/(2*s2)),exp(-D2Y/(2*s2))];
            
            %% akmmd-l2
            eta1=calculate_eta_l2(A,nX,nY);
            etastore=zeros(1,numperm);
            for iboot=1:numperm
                idx=randperm(nX+nY);
                etastore(iboot)=calculate_eta_l2(A(:,idx),nX,nY);
            end
            talp=quantile(etastore,1-alp);
            
            vote_m(irow,icol,irun)=double(eta1>talp);
            
            %% akmmd-spec
            [~,~,v]=svd(A,'econ');
            eta1=calculate_eta_spec(v,nX,nY,targetspec);
            etastore=zeros(1,numperm);
            for iboot=1:numperm
                idx=randperm(nX+nY);
                etastore(iboot)=calculate_eta_spec(v(idx,:),nX,nY,targetspec);
            end
            talp=quantile(etastore,1-alp);
            
            vote_mspec(irow,icol,irun)=double(eta1>talp);
            
        end
        
        %% KS-randproj
        data=cat(1,X,Y);
        
        % numrp many random projections
        us=zeros(dim,1,numrp);
        for ii=1:numrp
            us(:,:,ii)=svd(randn(dim,1),'econ');
        end
        
     
        eta1=calculate_KSrandpj(data,nX,nY,us);
         
        etastore=zeros(1,numperm);
        for iboot=1:numperm
            idx=randperm(nX+nY);
            etastore(iboot)=calculate_KSrandpj(data(idx,:),nX,nY,us);
        end
        
        talp=quantile(etastore,1-alp);
       
        vote_ks(1,icol,irun)=double(eta1>talp);
        
    end
    fprintf('\n')
end
fprintf('\n')

%% compute type I and type II error
%
powg=zeros(nrow,ncol);
powm=zeros(nrow,ncol);
powmspec=zeros(nrow,ncol); 
powks=zeros(1,ncol); 

for icol=1:ncol
    tmp=reshape(vote_g(:,icol,:),nrow,nrun);
    powg(:,icol)=sum( tmp,2)/nrun;
    tmp=reshape(vote_m(:,icol,:),nrow,nrun);
    powm(:,icol)=sum( tmp,2)/nrun;
    tmp=reshape(vote_mspec(:,icol,:),nrow,nrun);
    powmspec(:,icol)=sum( tmp,2)/nrun;
    
    tmp=reshape(vote_ks(1,icol,:),1,nrun);
    powks(:,icol)=sum( tmp,2)/nrun;
end

disp('-- Gmmd --')
disp(powg*100)

disp('-- Mmmd --')
disp(powm*100)

disp('-- Mmmd spec --')
disp(powmspec*100)

disp('-- KS randproj --')
disp(powks*100)


%%
irow1=3;
irow2=3;
irow3=3;

figure(22),clf; hold on;
plot(del_list,powg','x--b');
plot(del_list,powks,'x--g');
plot(del_list,powm(irow2,:),'x-r');
plot(del_list,powmspec(irow3,:),'x-m');
grid on;
xlabel('delta');title('power')


%% type I and II error with errorbar

% bootstrap to obtain errorbar of power

nboot1=40;
nrun1=floor(nrun/2);

pows_g=zeros(nrow,ncol,nboot1);
pows_m=zeros(nrow,ncol,nboot1);
pows_mspec=zeros(nrow,ncol,nboot1);
pows_ks=zeros(1,ncol,nboot1);

for iboot=1:nboot1
    idx=randperm(nrun,nrun1);
    
    tmp=vote_g(:,:,idx);
    pows_g(:,:,iboot)=mean(tmp,3);
    
    tmp=vote_m(:,:,idx);
    pows_m(:,:,iboot)=mean(tmp,3);
    
    tmp=vote_mspec(:,:,idx);
    pows_mspec(:,:,iboot)=mean(tmp,3);
    
    tmp=vote_ks(1,:,idx);
    pows_ks(1,:,iboot)=mean(tmp,3);
    
end

figure(23),clf; hold on;
%
p=reshape(pows_g(irow1,:,:),ncol,nboot1);
errorbar(del_list,mean(p,2),std(p,1,2),'x--b');
%
p=reshape(pows_ks(1,:,:),ncol,nboot1);
errorbar(del_list,mean(p,2),std(p,1,2),'x--g');
%
p=reshape(pows_m(irow2,:,:),ncol,nboot1);
errorbar(del_list,mean(p,2),std(p,1,2),'x-r');
%
p=reshape(pows_mspec(irow3,:,:),ncol,nboot1);
errorbar(del_list,mean(p,2),std(p,1,2),'x-m');
axis([0,delta,0,1]);
xlabel('\delta');ylabel('rejection rate')
grid on;
legend({'Gaussian MMD', 'Random Proj KS', 'Anisotropic L2', 'Anisotropic Spec'})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% witness function 

%%
ntest=400;
nX=ntest;
nY=ntest;

delta=delta_max;
[X,Y]=generate_curve_data(ntest,delta,epsx);

%% grid of x to visualize

nl=256;
[xx,yy]=meshgrid(1:1:nl, 1:1:nl);
xx=xx/nl*1.2;
yy=yy/nl*1.2;
xx=[xx(:),yy(:)];

%% gaussian kernel
D2=pdist2(xx,cat(1,X,Y)).^2;
s2=sig2g*sig2_list(irow1);
K1=exp(-D2/(2*s2));
w1= mean(K1(:,1:nX),2) - mean(K1(:,nX+1:nX+nY),2);

figure(31),clf;
scatter(xx(:,1),xx(:,2),80,w1,'o','filled')
colorbar();grid on;
title('witness gaussian kernel')


%% ak L2

m2X=gmR.mahal(X)';
m2Y=gmR.mahal(Y)';
m2xx=gmR.mahal(xx)';

s2=sig2m*sig2_list(irow2);
A=[exp(-m2X/(2*s2)),exp(-m2Y/(2*s2))];
Axx=exp(-m2xx/(2*s2));

w2= (Axx'*(mean(A(:,1:nX),2) - mean(A(:,nX+1:nX+nY),2)))/nR;


figure(32),clf;
scatter(xx(:,1),xx(:,2),80,w2,'o','filled')
colorbar();grid on;
title('witness ak l2')


%% ak spec
[u,s,v]=svd(A,'econ');
vX=v(1:nX,:)';
vY=v(nX+1:nX+nY,:)';
vxx= diag(1./diag(s))*(u'*Axx);

ltrunc=sum(targetspec>1e-10);

w3=vxx(1:ltrunc,:)'*( ...
    targetspec(1:ltrunc).*(mean(vX(1:ltrunc,:),2)-mean(vY(1:ltrunc,:),2)));


figure(33),clf;
scatter(xx(:,1),xx(:,2),80,w3,'o','filled')
colorbar();grid on;
title('witness ak spec')

return 
end




function [x,y]=generate_curve_data(n,delta,epsx)

tx=sort(rand(n,1));
x=[cos(tx*pi/2),sin(tx*pi/2)];
x=x+randn(size(x))*epsx;

ty=sort(rand(n,1));

%ry=1+delta*sin(ty*pi/2*6);

ry=1+delta*ones(size(ty));
y=diag(ry)*[cos(ty*pi/2),sin(ty*pi/2)];


y=y+randn(size(y))*epsx;


end





function [R,gmR]=generate_ref_local_pca(nR,funcXY,sig2g)

%% get a pool of X and Y
npool=1000;
kNN=40; %proportional to npool

[x,y]=funcXY(npool);

dim=size(x,2);

data=cat(1,x,y);
xdata=data(1:npool,:);
ydata=data(npool+1:npool*2,:);

%% 
R=generate_uniform_reference_set(data,nR,kNN);

%% sigma_r by local pca

k_loccov= floor(2*npool/4);

reg_cov=0.01.^2;

tic,
[idxnb,~]=knnsearch(data,R,'k',k_loccov);
toc

SR=zeros(dim,dim,nR);
for iR=1:nR
    xi=data(idxnb(iR,:),:);
    
    C=cov(xi);
    
    [v,d]=eig(C);
    [d,tmp]=sort(diag(d),'descend');
    v=v(:,tmp);
    d(d<reg_cov)=reg_cov;
    C=v*diag(d)*v';
    C=(C+C')/2;
    
    SR(:,:,iR)=C;
end

%% rescale the cov by a constant
l1s=zeros(nR,1);
for ir=1:nR
    C=SR(:,:,ir);
    l1s(ir)=min(eig(C));
end

c=median(l1s)/sig2g;
SR=SR/c;

%%
gmR=gmdistribution(R,SR,ones(1,nR)/nR);


%% vis the kernel
ARX=exp(-gmR.mahal(xdata)/2)';
ARY=exp(-gmR.mahal(ydata)/2)';

hX=mean(ARX,2);
hY=mean(ARY,2);% deviation of q from p

%% vis
if (1)
     
    n1=200;
    figure(1),clf;hold on;
    tmp=randperm(npool,n1);
    scatter(xdata(tmp,1),xdata(tmp,2),80,'.g')
    scatter(ydata(tmp,1),ydata(tmp,2),80,'.b')
    grid on; axis equal
    title('data X Y')
    
    figure(2),clf;
    scatter(R(:,1),R(:,2),80,hX-hY,'o','filled');
    grid on; colorbar();title('hX-hY');
    
    %
    iR=floor(nR*.75);
    figure(3),clf,hold on;
    scatter(ydata(:,1),ydata(:,2),60,ARY(iR,:),'o')
    scatter(xdata(:,1),xdata(:,2),60,ARX(iR,:),'o')
    scatter(R(iR,1),R(iR,2),'xr');
    axis equal;grid on; 
    title('A(r,x)');
    drawnow();
end


end


function [D2]=euclidean_dis2(X,Y)
dis2XX=squareform(pdist(X).^2);
dis2YY=squareform(pdist(Y).^2);
dis2XY=pdist2(X,Y).^2;
D2=[dis2XX, dis2XY; dis2XY', dis2YY];
end


function [m2X,m2Y]=mahal_dis2(X,Y,gm)
m2X=gm.mahal(X)';
m2Y=gm.mahal(Y)';
end


function eta=calculate_eta_l2(A,nX,nY)
hX=mean(A(:,1:nX),2);
hY=mean(A(:,nX+1:nX+nY),2);
eta=mean((hX-hY).^2);
end

function eta=calculate_eta_spec(v,nX,nY,targetspec)

num1=min(size(v,2),numel(targetspec));

vX=v(1:nX,1:num1);
vY=v(nX+1:nX+nY,1:num1);
vvX=mean(vX,1)';
vvY=mean(vY,1)';

eta=sum((vvX-vvY).^2.*targetspec(1:num1));
end

function eta=calculate_kernel_mmd2(K,nX,nY)
assert(size(K,1)==nX+nY);
KXX=K(1:nX,1:nX);
KXY=K(1:nX,nX+1:nX+nY);
KYY=K(nX+1:nX+nY,nX+1:nX+nY);
eta=mean(KXX(:))+ mean(KYY(:))-2*mean(KXY(:));
end

function dd=calculate_KSrandpj(data,nX,nY,us)

[n,dim]=size(data);
assert(n==nX+nY);
assert(size(us,1)==dim)
numrp=size(us,3);

%%
dd=0;
for ii=1:numrp
    u=us(:,:,ii);
    
    d1=data*u;
    
    X1=d1(1:nX);
    Y1=d1(nX+1:nX+nY,:);
    
    dd=dd+compute_KSstat(X1,Y1);
end
end


function KSstatistic = compute_KSstat(x1, x2)

%
% Calculate F1(x) and F2(x), the empirical (i.e., sample) CDFs.
%

binEdges    =  [-inf ; sort([x1;x2]) ; inf];

binCounts1  =  histc (x1 , binEdges, 1);
binCounts2  =  histc (x2 , binEdges, 1);

sumCounts1  =  cumsum(binCounts1)./sum(binCounts1);
sumCounts2  =  cumsum(binCounts2)./sum(binCounts2);

sampleCDF1  =  sumCounts1(1:end-1);
sampleCDF2  =  sumCounts2(1:end-1);

%
% Compute the test statistic of interest.
%

%  2-sided test: T = max|F1(x) - F2(x)|.
deltaCDF  =  abs(sampleCDF1 - sampleCDF2);

KSstatistic   =  max(deltaCDF);
end
