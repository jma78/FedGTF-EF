%% Author: Jing Ma, Emory University
function [X, Xs, dim, nonzero_ratio, cutoffs] = genTensor(K, fileName,isBinary)
% X: observered big tensor
% Xs: cell of observered small tensor (each tensor is hold by one hospital)
% dim: dimension of observered big tensor
% nonzero_ratio: Non zero elements ratio of observered big tensor
% K: number of hopsital (number of small tensor)
% cutoffs: divide number of the first dimension of big tensor (patient mode)
%% load data
max_count = 1; % threshold of count (elements' value of tensor)
skewness = 0.6;
% Read MIMICIII dataset
% fileName=strcat('cms1k.csv');
count_mat=csvread(fileName); % count, patient, diag, proc, med
count_mat=double(count_mat);
if isBinary
    count_mat((count_mat(:,1)>max_count),1)=max_count;
end
sz=max(count_mat); % size of the tensor
numP=sz(2); % the largest patient ID
dim = sz(2:end);

%% when data unevenly partitioned, get the number of each group
patient_unique = unique(count_mat(:,2)); % number of patients
numP_unique = length(patient_unique);
Xs_num = numP_unique/K;
patient_unique_perm = patient_unique(randperm(size(patient_unique, 1)), :);
edges = round(linspace(1,numP_unique+1, K+1));

%% cutoff rows for each ICU
cutoffs = cell(1,K);
for k=1:K
    cutoffs{k}=patient_unique_perm(edges(k):edges(k+1)-1);
end

%% get the small tensors
sum_tensors = cell(1,k);
for k=1:K
    sum_tensors{k} = count_mat(ismember(count_mat(:,2), cutoffs{k}),1:end);
end

%% Sparse tensor
% Sparse tensor of full data
X=sptensor(count_mat(:,2:end), count_mat(:, 1), sz(2:end));
nonzero_ratio = size(X.subs,1)/prod(size(X));
Xs = cell(1,K);
for k=1:K
    Xs{k} = sptensor(sum_tensors{k}(:, 2:end), sum_tensors{k}(:,1), sz(2:end));
end