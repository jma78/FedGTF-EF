clear all;clc;
warning('off');
addpath('tensor_toolbox-v3.1');
%% gendata

addpath(genpath('./data'));
K = 8; % number of clients
file = 'cms1k_dense';
fileName = strcat(file, '.csv');
isBinary = 1;  % 1 for logit loss, 0 for square loss
[X, Xs, dim, nonzero_ratio, cutoffs] = genTensor(K, fileName,isBinary);

rank = 10;

%% initialize
nd = ndims(X);
sz = size(X);
dim = size(X);

%% initilize each client

for k = 1:K
    client(k).X = Xs{k};
    sz = size(Xs{k});
    tsz = prod(sz);
    nmissing = 0;
    nnonzeros = nnz(Xs{k});
    nzeros = tsz - nnonzeros;    
    % Save info
    client(k).info.size = sz;
    client(k).info.tsz = tsz;
    client(k).info.nmissing = nmissing;
    client(k).info.nnonzeros = nnonzeros;
    client(k).info.nzeros = nzeros;
    client(k).EF{1} = zeros(sz(1),rank);
    for n = 2:nd
        client(k).EF{n} = zeros(sz(n),rank);
    end
end

Uinit = cell(1,nd); % Initial Global factor matrix
for n = 1:nd % n=1 initialize at institutions
    Uinit{n} = rand(dim(n),rank);
end
M0 = ktensor(Uinit);
M0 = M0 * (norm(X)/norm(M0)); % normalize
M0 = normalize(M0,0);
U = cell(1,nd);
for d=1:nd
    U{d} = M0{d};
end

for k=1:K
    client(k).dim=size(client(k).X); %data
    client(k).U=cell(1,nd); % initialize 3 factor matrices
    % n = 1
    client(k).U{1}=zeros(dim(1),rank);
    client(k).U{1}(cutoffs{k},:) = U{1}(cutoffs{k},:);
    for d=2:nd
        client(k).U{d}=U{d};
    end
end

oversample = 1.1;
for k=1:K
    xnzidx = tt_sub2ind64(size(client(k).X),client(k).X.subs);
    xnzidx = sort(xnzidx);
    tsz = prod(size(client(k).X));
    nnonzeros = nnz(client(k).X);
    nzeros = tsz - nnonzeros; 
    ftmp = max(ceil(nnz(client(k).X)/100), 10^5);
    fsamp(1) = min(ftmp, nnonzeros);
    fsamp(2) = min([ftmp, nnonzeros, nzeros]);
    [fsubs, fvals, fwgts] = tt_sample_stratified(client(k).X, xnzidx, fsamp(1), fsamp(2), oversample);
    client(k).fsubs = fsubs;
    client(k).fvals = fvals;
    client(k).fwgts = fwgts;
end

%aggregate the sampled entries
subs = [];
vals = [];
wgts = [];
for k=1:K
   subs = vertcat(subs, client(k).fsubs);
   vals = vertcat(vals, client(k).fvals);
   wgts = vertcat(wgts, client(k).fwgts);
end


%%%%%%%%%%%%%%%%%%%% Run Logit %%%%%%%%%%%%%%%%%%%%
%% initialize
rank = 10;
params.maxepoch = 500;
params.epciters = 50;
params.nsamplsq = 5;
nd = ndims(X);
gsamp_rate = 10;
prox = 1e-4;  
isLogit = 1;  % set to 1 for logit loss, 0 for square loss

%%%%%%%%% FedGTF-EF-PC %%%%%%%%%%%%%
params.epciters = 50;
params.maxepoch = 500;
tau = 8;
lr= 2^-1;
isCyclic = 0;
G = FedGTF_EF_PC(client,U,nd,K,dim,rank,subs,vals,wgts,isLogit,isCyclic,params,lr,tau);
