function [G] = gcp_fg_est_G_rd(U, fh, gh, subs, xvals,d,sz,rd)
%GCP_FG_EST Estimate the GCP function and gradient with a subsample
%
%   [F,G] = GCP_FG_EST(M, FH, GH, XSUBS, XVALS, WVALS) estimates the GCP
%   function and gradient specified by FH and GH for M and X. In this case,
%   we have only a portion of X as  specified by XSUBS and XVALS along with
%   the corresponding sampling weights in WVALS that are used in the estimate.
%
%   See also GCP_SGD, GCP_FG, GCP_FG_SETUP.
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.

% Created by Tamara G. Kolda, Fall 2018. Includes work with
% collaborators David Hong and Jed Duersch. 

%% Hidden options
%
%  Note that there are five hidden options. The first three are similar to
%  the hidden options for gcp_fg and the fourth is whether or not to verify
%  that M has lambda = [1,1,...,1], which is assumed and so should
%  generally be checked unless the user is absolutely sure it's okay. 

%% Parse inputs

%% Input checks (keep minimal for timing's sake)

% d = ndims(M);
% sz = size(M);

G = [];

% if LambdaCheck && ~all(M.lambda == 1)
%     warning('Fixing M to have all ones for lambda');
%     M = normalize(M,1);
% end

%% Compute model values and exploded Zk matrices
[mvals, Zexp] = gcp_fg_est_helper(U, subs);

%% Compute function value

%% Compute sample y values
yvals = gh(xvals, mvals);

%% Compute function and gradient
nsamples = size(subs,1);
  
    S = sparse(subs(:,rd), (1:nsamples)', yvals, sz(rd), nsamples, nsamples);    
    G = S * Zexp{rd};


function [mvals, Zexp] = gcp_fg_est_helper(factors, subs)
% GCP_FG_EST_HELPER Model values at sample locations and exploded Zk's.  

% Created by Tamara G. Kolda, Sept. 2018. Includes prior work by
% collaborators David Hong and Jed Duersch.  

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
% for k = 2,...,d.
Zexp = cell(1,d);
Zexp{2} = Uexp{1};
for k = 3:d
    Zexp{k} = Zexp{k-1} .* Uexp{k-1};
end

% After this pass,
% Zexp{k} = Hadamard product of Uexp{1} though Uexp{d}, except Uexp{k}
% for k = 1,...,d.
Zexp{1} = Uexp{d};
for k=d-1:-1:2
    Zexp{k} = Zexp{k} .* Zexp{1};
    Zexp{1} = Zexp{1} .* Uexp{k};
end

% Compute model values at sample locations
mvals = sum(Zexp{d} .* Uexp{d},2);


