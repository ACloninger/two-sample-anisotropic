
    % You are free to use, change, or redistribute this code in any way you
    % want for non-commercial purposes. However, it is appreciated if you 
    % maintain the name of the original author, and cite the paper:
    % X. Cheng, A. Cloninger, R. Coifman.  "Two Sample Statistics Based on Anisotropic Kernels."
    % arxiv:1709.05006
    %
    % Date: October 20, 2017. (Last Modified: October 20, 2017)

function R=generate_uniform_reference_set(data,nR,kNN)

% input:
%    data   [n,dim]
%    nR     output R is [nR,dim]
%    kNN    number of nearest neighbors to estimate epsdata1 and epsdata, 
%           every point has a scale which is its median distance to its kNN
%           neighbors. epsdata1 is the mean of this scale over dataset,
%           epsdata is the max of this scale. epsdata1 is seen as the
%           "smallest scale" in the dataset, and used in the kernel for kde
%           for sampling ref, and in pruning the dataset; epsdata is used
%           to remove outlier ref point after initial sampling.

[n,dim]=size(data);

%% estimate epsdata1 in data

n1=min(n,1e3);
tic,
[~,dd]=knnsearch(data,data(randperm(n,n1),:),'k',kNN);
toc

%
dis1= dd(:,kNN);
epsdata1=median(dis1) %smallest timescale in data for kde

%
epsdata=quantile(dis1,.99) %to prune ref est R if mdis is larger than epsdata


%%
maxnumbatch=100;

nbatch=min(n,1000);
nsample=floor(nbatch/10);

i=0;
R=[];
for ibatch=1:maxnumbatch
    
    data1=data(randperm(n,nbatch),:);
    tic
    [~,d1]=knnsearch(data,data1,'k',kNN);
    toc
    
    % kde on data
    aff=exp(-d1.^2/(2*(epsdata1)^2));
    nu=sum(aff,2);
    p=1./nu;
    p=p/sum(p);
    
    % sample from data
    r=mnrnd(nsample,p);
    xi=data1(r>0,:); 

    xi=xi+(rand(size(xi))-.5)*((epsdata1)/sqrt(dim)); %giggering

    % pruning the set
    if size(R,1)>0
        dis=min(pdist2(xi,R),[],2);
        xi( dis<epsdata1,:)=[];
    end
    nxi=size(xi,1);
     
    x=[];
    for ii=1:nxi
        nn=size(xi,1);
        if nn<1
            break;
        end
        isel=randperm(nn,1);
        x=[x;xi(isel,:)];
        dis=pdist2(xi,xi(isel,:));
        xi( dis<epsdata1,:)=[];
    end

    if size(x,1)<1
        continue;
    end

    % exclude too faraway points
    [~,dis]=knnsearch(data,x,'k',kNN);
    disknn=dis(:,kNN); 
    idx=find(disknn<epsdata);
    ni=numel(idx);
    R=[R;x(idx,:)];
    i=i+ni;
    
    if i>nR
        break;
    end
end

ibatch
if ibatch==maxnumbatch
    warning('max number of batch reached.')
end

%%
nR1=size(R,1);
R=R(randperm(nR1),:);
if nR1>nR
    R=R(1:nR,:);
end


[~,tmp]=sort(R(:,1),'ascend');
R=R(tmp,:);

end
