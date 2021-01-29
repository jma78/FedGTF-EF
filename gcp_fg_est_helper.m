function [mvals, Zexp] = gcp_fg_est_helper(factors, subs)
% GCP_FG_EST_HELPER Model values at sample locations and exploded Zk's.   

% Check for empty
if isempty(subs)
    mvals = [];
    return;
end

% Process inputs
d = size(subs,2);

% Create exploded U's from the model factor matrices
Uexp = cell(d,1);
for k = 1:d
    Uexp{k} = factors{k}(subs(:,k),:);
end

% After this pass,
% Zexp{k} = Hadarmard product of Uexp{1} through Uexp{k-1}
Zexp = cell(1,d);
Zexp{2} = Uexp{1};
for k = 3:d
    Zexp{k} = Zexp{k-1} .* Uexp{k-1};
end

% After this pass,
% Zexp{k} = Hadamard product of Uexp{1} though Uexp{d}, except Uexp{k}
Zexp{1} = Uexp{d};
for k=d-1:-1:2
    Zexp{k} = Zexp{k} .* Zexp{1};
    Zexp{1} = Zexp{1} .* Uexp{k};
end

% Compute model values at sample locations
mvals = sum(Zexp{d} .* Uexp{d},2);
