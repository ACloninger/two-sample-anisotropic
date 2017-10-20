% set number of Monte Carlo runs at line 343


function demo_sec3_limiting_density()

clear all; close all;
rng(9112017);


%% p and q parameter
dim=2;
epsx=0.02;
delta=0.02;

%% construnct r and sigma_r
nR=100; 

%
npool=4000;
kNN=40; %proportional to npool

[x,y]=generate_curve_data(npool,delta,epsx);
data=cat(1,x,y);
xdata=data(1:npool,:);
ydata=data(npool+1:npool*2,:);

% reference set
R=generate_uniform_reference_set(data,nR,kNN);

% vis
n1=200;
figure(1),clf;hold on;
tmp=randperm(npool,n1);
scatter(xdata(tmp,1),xdata(tmp,2),80,'og')
scatter(ydata(tmp,1),ydata(tmp,2),80,'xb')
grid on; drawnow();axis equal
title('data X Y')


%% covariance field Sigma_r

% 
C=diag(([epsx,epsx]).^2);
gmRg=gmdistribution(R,C,ones(1,nR)/nR);

%
sig_Lambda1=0.2;
Lambda1=diag(([sig_Lambda1,epsx]).^2);
SR=zeros(dim,dim,nR);
for iR=1:nR
    
    ri=R(iR,:);
    phi1=[ri(2),-ri(1)]';
    phi1=phi1/norm(phi1);
    phi2=[ri(1),ri(2)]';
    phi2=phi2/norm(phi2);
    C=[phi1,phi2]*Lambda1*[phi1,phi2]';
    C=C/2;                              %because K=A*A^T
    SR(:,:,iR)=C;
end
gmRm=gmdistribution(R,SR,ones(1,nR)/nR);

%
if (1)
    gmR=gmRm;
else
    gmR = gmRg; %use isotropic gaussian
end

%% vis the kernel
ARX=exp(-gmR.mahal(xdata)/2)';
ARY=exp(-gmR.mahal(ydata)/2)';

hX=mean(ARX,2);
hY=mean(ARY,2);% deviation of q from p

%
figure(2),clf;
scatter(R(:,1),R(:,2),80,hX-hY,'o','filled');
grid on; colorbar();title('hX-hY')

%
iR=floor(nR/4);
figure(3),hold on;
scatter(ydata(:,1),ydata(:,2),60,ARY(iR,:),'o')
scatter(xdata(:,1),xdata(:,2),60,ARX(iR,:),'o')
scatter(R(iR,1),R(iR,2),'xr'); 
axis equal;grid on;title('A(r,x)')

drawnow(); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% lambda_k and c_k
%%


%% compute spectrum of kernel on a large sample of p and q

n=5000;

[X,Y]=generate_curve_data(n,delta,epsx);

numspec=500;

%% gausian kernel
sig2g=(epsx)^2;

% sparse kernel by nearest neighbor
kNN=1000;
[idx,dis]=knnsearch(X,X,'k',kNN);
I=repmat((1:n)',[1,kNN]);
J=idx;
V=exp(-dis.^2/(2*sig2g));
Kx=sparse(I,J,V,n,n);


% centered kernel
K1x=mean(Kx,2);
K0x=mean(K1x);
Ktil=Kx-K1x*ones(n,1)'-ones(n,1)*K1x'+K0x*ones(n,n); 
Ktil=(Ktil+Ktil')/2;

% eig of Ktil
disp('... computing eigs of kernel matrix ...')
tic,
[v,d]=eigs(Ktil/n,numspec);
toc


[d,tmp]=sort(diag(d),'descend');
lambda1=d(1:numspec);
v=v(:,tmp);
psi1x=v(:,1:numspec)*sqrt(n);

% nystrom to Y
[idx,dis]=knnsearch(Y,X,'k',kNN);
I=repmat((1:n)',[1,kNN]);
J=idx;
V=exp(-dis.^2/(2*sig2g));
Kxy=sparse(I,J,V,n,n);

%center
K1y=mean(Kxy,1)';
K0y=mean(K1y);
Ktil=Kxy-K1x*ones(n,1)'-ones(n,1)*K1y'+K0x*ones(n,n);

psi1y=(Ktil'*psi1x/n)*diag(1./lambda1);

%ck
ck1=mean(psi1y-psi1x,1)';

if (0)
    % vis
    j=34;
    figure(10),hold on;
    scatter(X(:,1),X(:,2),40,psi1x(:,j),'o');
    scatter(Y(:,1),Y(:,2),40,psi1y(:,j),'o','filled');
    axis equal;colorbar();grid on;
    title(sprintf('j=%d, gaussian kernel',j))
    
    
    figure(11),clf;hold on;
    plot(ck1.^2.*lambda1,'.-');
    grid on; title('ck gaussian kernel')
    
    figure(12),hold on;
    plot(lambda1,'.-');
    grid on; title('lambdak gaussian kernel')
end

%% K_L2
[D2X,D2Y]=mahal_dis2(X,Y,gmR);

sig2m=1;
ARX=exp(-D2X/(2*sig2m));
ARY=exp(-D2Y/(2*sig2m));

%
A1=mean(ARX,2); % 
Atil=ARX-A1*ones(n,1)';

% compute eigenvalues \tilde{\lambda_k} which are w.r.t p 
[u,s,v]=svd(Atil/sqrt(n*nR),'econ');
s=diag(s);
numk=min(numspec,nR);
lambda2=s(1:numk).^2;
psi2x=v(:,1:numk)*sqrt(n);

% nystrom
Ktil=(ARX-A1*ones(n,1)')'*(ARY-A1*ones(n,1)')/nR;

psi2y=(Ktil'*psi2x/n)*diag(1./lambda2);

%ck
ck2=mean(psi2y-psi2x,1)';


if (0)
    %% vis
    j=5;
    figure(20),hold on;
    scatter(X(:,1),X(:,2),40,psi2x(:,j),'o');
    scatter(Y(:,1),Y(:,2),40,psi2y(:,j),'o','filled');
    axis equal;colorbar();grid on;
    title(sprintf('j=%d, KL2 kernel',j))
    
    
    
    figure(21),clf;hold on;
    %plot(abs(ck2),'.-');
    plot(ck2.^2.*lambda2,'.-');
    grid on; title('ck KL2 kernel')
    
    figure(22),hold on;
    plot(lambda2,'.-');
    grid on; title('lambdak KL2 kernel')
end

%% K_spec

lspec=20;

ll=(1:numspec)';
targetspec=exp(-2*(ll-(lspec-4)))./(exp(-2*(ll-(lspec-4)))+1);


%%
A=[ARX,ARY];

[~,~,vA]=svd(A,'econ');
p3x=vA(1:n,:)*sqrt(2*n);
p3y=vA(n+1:n*2,:)*sqrt(2*n);

%
num1=min(size(p3x,2),numspec);
ax=diag(sqrt(targetspec(1:num1)))*p3x(:,1:num1)';

A1=mean(ax,2);
Atil=ax-A1*ones(n,1)';

%
[u,s,v]=svd(Atil/sqrt(n),'econ');
s=diag(s);


ltrunc=sum(s>1e-10); 
psi3x=v(:,1:ltrunc)*sqrt(n);
lambda3=s(1:ltrunc).^2;


%
ay=diag(sqrt(targetspec(1:num1)))*p3y(:,1:num1)';
Ktil=(ax-A1*ones(n,1)')'*(ay-A1*ones(n,1)');

psi3y=(Ktil'*psi3x/n)*diag(1./lambda3);

%ck
ck3=mean(psi3y-psi3x,1)';

if (0)
    %% vis
    j=5;
    
    figure(30),clf;hold on;
    scatter(X(:,1),X(:,2),40,psi3x(:,j),'o');
    scatter(Y(:,1),Y(:,2),40,psi3y(:,j),'o','filled');
    axis equal;colorbar();grid on;
    title(sprintf('j=%d, Kspec kernel',j))
    
    figure(31),clf;hold on;
    plot(ck3.^2.*lambda3,'.-');
    grid on; title('ck Kspec kernel')
    
    figure(32),clf,hold on;
    plot(lambda3,'.-');
    grid on; title('lambdak Kspec kernel')
end

%% constant to rescale T
ll1=lambda1(1);
ll2=lambda2(1);
ll3=lambda3(1);


%%
% notice lambda3 is not the target spec: lambda3 is the \tilde{lambda}
% after centering the kernel
numspecvis=50;
num3=min(ltrunc,numspecvis);

figure(41),clf; hold on;
plot(lambda1(1:numspecvis)/ll1,'s-b');
plot(lambda2(1:numspecvis)/ll2,'x-r');
plot(lambda3(1:num3)/ll3,'o-k');
grid on;xlabel('k')
%title('\tilde \lambda_k')
title('eigenvalues \lambda_k of centered kernel')
legend('gaussian', 'k_{L^2}', 'k_{spec}')


%
figure(42),clf; hold on;
stem(ck1(1:numspecvis).^2.*(lambda1(1:numspecvis)/ll1),'s-b');
stem(ck2(1:numspecvis).^2.*(lambda2(1:numspecvis)/ll2),'x-r');
stem(ck3(1:num3).^2.*(lambda3(1:num3)/ll3),'o-k');
grid on;xlabel('k')
%title('\tilde{\lambda}_k c_k^2')
title('\lambda_k c_k^2 of centered kernel')
legend('gaussian', 'k_{L^2}', 'k_{spec}')




drawnow();

%%%%%%%%%%%%%d%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% empirical distribution of Tn
%%


%% two sample test

if (1)
    % case 1
    ntest= 200;
    tau= 0.5;
    
else
    % case 2
    ntest= 400;
    tau= 0.5/sqrt(2);
end

%% new q = tau*q + (1-tau)*p
nq=floor(ntest*tau);
np=ntest-nq;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  set the number of Monte Carlo runs
%

nrun = 1000;

%nrun = 10000;
    % to reproduce figures in the paper, use nrun= 10000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
mmd1=zeros(2,nrun);
mmd2=zeros(2,nrun);
mmd3=zeros(2,nrun);
nX=ntest;
nY=ntest;

for irun=1:nrun
     
    if mod(irun,10)==0
        fprintf('-%d-',irun)
    end
    
    %%%%%%%%
    %% pq
    [x1,~]=generate_curve_data(ntest,delta,epsx);
    [x2,y2]=generate_curve_data(ntest,delta,epsx);
    X=x1;
    Y=cat(1,x2(randperm(ntest,np),:),y2(randperm(ntest,nq),:));
    
    %% gaussian kernel
    dis2XX=squareform(pdist(X).^2);
    dis2YY=squareform(pdist(Y).^2);
    dis2XY=pdist2(X,Y).^2;
    
    H=exp(-dis2XX/(2*sig2g))+exp(-dis2YY/(2*sig2g))...
      -exp(-dis2XY/(2*sig2g))-exp(-dis2XY/(2*sig2g))';
    
    mmd1(1,irun)=mean(H(:));
    
    %% K_L2
    [D2X,D2Y]=mahal_dis2(X,Y,gmR);
    A=[exp(-D2X/(2*sig2m)),exp(-D2Y/(2*sig2m))];
    mmd2(1,irun)=calculate_eta_l2(A,nX,nY);
    
    %% K_spec
    [~,~,v]=svd(A,'econ');
    v=v*sqrt(ntest*2);
    
    mmd3(1,irun)=calculate_eta_spec(v,nX,nY,targetspec);
    
    %%%%%%%%
    %% pp
    [x1,~]=generate_curve_data(ntest,delta,epsx);
    [x2,~]=generate_curve_data(ntest,delta,epsx);
    X=x1;
    Y=x2;
    
    %% gaussian kernel
    dis2XX=squareform(pdist(X).^2);
    dis2YY=squareform(pdist(Y).^2);
    dis2XY=pdist2(X,Y).^2;
    
    H=exp(-dis2XX/(2*sig2g))+exp(-dis2YY/(2*sig2g))...
      -exp(-dis2XY/(2*sig2g))-exp(-dis2XY/(2*sig2g))';
    
    mmd1(2,irun)=mean(H(:));
    
    %% K_L2
    [D2X,D2Y]=mahal_dis2(X,Y,gmR);
    A=[exp(-D2X/(2*sig2m)),exp(-D2Y/(2*sig2m))];
    mmd2(2,irun)=calculate_eta_l2(A,nX,nY);
    
    %% K_spec
    [~,~,v]=svd(A,'econ');
    v=v*sqrt(ntest*2);
    
    mmd3(2,irun)=calculate_eta_spec(v,nX,nY,targetspec);
    
end
fprintf('\n')


%% plot histogram
numbin=40;

%
figure(51), clf;hold on;
l1=mmd1(2,:); 
l2=mmd1(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac1= sum(l2>quantile(l1,0.95))/n1
title('T under H0 and H1, gaussian mmd')


%
figure(52), clf;hold on;
l1=mmd2(2,:); 
l2=mmd2(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac1= sum(l2>quantile(l1,0.95))/n1
title('T under H0 and H1, kL2 mmd')

%
figure(53), clf;hold on;
l1=mmd3(2,:); 
l2=mmd3(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac1= sum(l2>quantile(l1,0.95))/n1
title('T under H0 and H1, kspec mmd')



%% asymptotic value by theory


nruna=50000;

mmd1a=zeros(2,nruna);
mmd2a=zeros(2,nruna);
mmd3a=zeros(2,nruna);

for irun=1:nruna
    
    mmd1a(1,irun)=sum(lambda1.*(...
          randn(numspec,1)*sqrt(2/ntest) + ...
        + (-tau)*ck1 ).^2 );
    
    mmd1a(2,irun)=sum(lambda1.*(...
        (randn(numspec,1)*sqrt(2/ntest)).^2));
    
    
    numk=min(numspec,nR);
    mmd2a(1,irun)=sum(lambda2.*(...
          randn(numk,1)*sqrt(2/ntest) + ...
        + (-tau)*ck2 ).^2 );
    
    mmd2a(2,irun)=sum(lambda2.*(...
        (randn(numk,1)*sqrt(2/ntest)).^2));
    
    
    mmd3a(1,irun)=sum(lambda3(1:ltrunc).*(...
          randn(ltrunc,1)*sqrt(2/ntest) + ...
        + (-tau)*ck3(1:ltrunc) ).^2 );
    
    mmd3a(2,irun)=sum(lambda3(1:ltrunc).*(...
         (randn(ltrunc,1)*sqrt(2/ntest)).^2));
    
end



%% rescale the T and the lambda
mmd1=mmd1/ll1;
mmd2=mmd2/ll2;
mmd3=mmd3/ll3;

mmd1a=mmd1a/ll1;
mmd2a=mmd2a/ll2;
mmd3a=mmd3a/ll3;


%%
numbin=40;

%
figure(51),clf; hold on;

l1=mmd1(2,:); 
l2=mmd1(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac1= sum(l2>quantile(l1,0.95))/n1

l1=mmd1a(2,:);
l2=mmd1a(1,:);
l1=sqrt(l1);
l2=sqrt(l2);


n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.--b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.--r');
frac11= sum(l2>quantile(l1,0.95))/n1
grid on;

title(sprintf('gaussian kernel, %4.2f %4.2f',frac1*100,frac11*100))

%
figure(52),clf; hold on;

l1=mmd2(2,:); 
l2=mmd2(1,:);
l1=sqrt(l1);
l2=sqrt(l2);


n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac2= sum(l2>quantile(l1,0.95))/n1

l1=mmd2a(2,:);
l2=mmd2a(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.--b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.--r');
frac21= sum(l2>quantile(l1,0.95))/n1
grid on;

title(sprintf('kL2, %4.2f %4.2f',frac2*100,frac21*100))


%
figure(53),clf; hold on;

l1=mmd3(2,:); 
l2=mmd3(1,:);
l1=sqrt(l1);
l2=sqrt(l2);


n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.-b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.-r');
frac2= sum(l2>quantile(l1,0.95))/n1

l1=mmd3a(2,:);
l2=mmd3a(1,:);
l1=sqrt(l1);
l2=sqrt(l2);

n1=numel(l1);n2=numel(l2);
[nout1,xout1]=hist(l1,numbin);
plot(xout1, (nout1/n1)/(xout1(2)-xout1(1)), '.--b');
[nout2,xout2]=hist(l2,numbin);
plot(xout2, (nout2/n2)/(xout2(2)-xout2(1)), '.--r');
frac21= sum(l2>quantile(l1,0.95))/n1
grid on;

title(sprintf('kspec, %4.2f %4.2f',frac2*100,frac21*100))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% collect values of theta0 etc in the table

theta0_1=mean(mmd1(2,:));
sigma0_1=std(mmd1(2,:));

theta0_2=mean(mmd2(2,:));
sigma0_2=std(mmd2(2,:));

theta0_3=mean(mmd3(2,:));
sigma0_3=std(mmd3(2,:));

bartheta0_1=mean(mmd1a(2,:));
barsigma0_1=std(mmd1a(2,:));

bartheta0_2=mean(mmd2a(2,:));
barsigma0_2=std(mmd2a(2,:));

bartheta0_3=mean(mmd3a(2,:));
barsigma0_3=std(mmd3a(2,:));


theta1_1=mean(mmd1(1,:));
sigma1_1=std(mmd1(1,:));

theta1_2=mean(mmd2(1,:));
sigma1_2=std(mmd2(1,:));

theta1_3=mean(mmd3(1,:));
sigma1_3=std(mmd3(1,:));

bartheta1_1=mean(mmd1a(1,:));
barsigma1_1=std(mmd1a(1,:));

bartheta1_2=mean(mmd2a(1,:));
barsigma1_2=std(mmd2a(1,:));

bartheta1_3=mean(mmd3a(1,:));
barsigma1_3=std(mmd3a(1,:));

r_1=(theta1_1-theta0_1)/(sigma1_1+sigma0_1);
r_2=(theta1_2-theta0_2)/(sigma1_2+sigma0_2);
r_3=(theta1_3-theta0_3)/(sigma1_3+sigma0_3);

r_1a=(bartheta1_1-bartheta0_1)/(barsigma1_1+barsigma0_1);
r_2a=(bartheta1_2-bartheta0_2)/(barsigma1_2+barsigma0_2);
r_3a=(bartheta1_3-bartheta0_3)/(barsigma1_3+barsigma0_3);


%
fprintf('--- theta0 \t theta1 \t sigma0 \t sigma1 \t r ----\n')
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        theta0_1,theta1_1,sigma0_1,sigma1_1,r_1 );
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        theta0_2,theta1_2,sigma0_2,sigma1_2,r_2 );
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        theta0_3,theta1_3,sigma0_3,sigma1_3,r_3 );


fprintf('--- by theory ----\n')
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        bartheta0_1,bartheta1_1,barsigma0_1,barsigma1_1,r_1a );
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        bartheta0_2,bartheta1_2,barsigma0_2,barsigma1_2,r_2a );
fprintf('%6.4f\t%6.4f\t%6.4f\t%6.4f\t%6.4f\t\n',...
        bartheta0_3,bartheta1_3,barsigma0_3,barsigma1_3,r_3a );
    
%%
% n=200
% --- theta0 	 theta1 	 sigma0 	 sigma1 	 r ----
% 0.4771	0.5444	0.0677	0.0704	0.4874	
% 0.0489	0.0958	0.0214	0.0306	0.9000	
% 0.0985	0.2046	0.0348	0.0587	1.1351	
% --- by theory ----
% 0.4754	0.5439	0.0676	0.0736	0.4848	
% 0.0488	0.0939	0.0214	0.0312	0.8573	
% 0.0983	0.2013	0.0374	0.0620	1.0354	

% % n=400
% --- theta0 	 theta1 	 sigma0 	 sigma1 	 r ----
% 0.2381	0.2720	0.0334	0.0359	0.4885	
% 0.0243	0.0477	0.0107	0.0153	0.8972	
% 0.0490	0.1036	0.0177	0.0305	1.1343	
% --- by theory ----
% 0.2379	0.2722	0.0339	0.0368	0.4850	
% 0.0244	0.0471	0.0106	0.0157	0.8616	
% 0.0490	0.1003	0.0188	0.0310	1.0290	

return;
end




function [x,y]=generate_curve_data(n,delta,epsx)

tx=sort(rand(n,1));
x=[cos(tx*pi/2),sin(tx*pi/2)];
x=x+randn(size(x))*epsx;

ty=sort(rand(n,1));
y=[cos(ty*pi/2),sin(ty*pi/2)];

y=y*(1-delta);
y=y+randn(size(y))*epsx;

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

