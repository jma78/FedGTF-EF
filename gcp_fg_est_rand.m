function [F, G] = gcp_fg_est_rand(M, fh, gh, subs, xvals, weights, computeF, computeG, vectorG, LambdaCheck, crng)
%% Parse inputs
if nargin < 11
       
    if ~exist('computeF','var')
        computeF = true;
    end
    
    if ~exist('computeG','var') 
        computeG = (nargout > 1);
    end
    
    if ~exist('vectorG','var')
        vectorG = false;
    end
    
    if ~exist('LambdaCheck','var')
        LambdaCheck = true;
    end
    
    % Specify range for correction/adjustment when nonzeros may be included
    % in the "zero" sample. In this case, crng should be the indices
    % of nonzero samples, which are the ones that are adjusted. 
    if ~exist('idx','var')
        crng = [];
    end
    
end

%% Input checks (keep minimal for timing's sake)

d = ndims(M);
sz = size(M);
F = [];
G = [];

if LambdaCheck && ~all(M.lambda == 1)
    warning('Fixing M to have all ones for lambda');
    M = normalize(M,1);
end

%% Compute model values and exploded Zk matrices
[mvals, Zexp] = gcp_fg_est_helper(M.u, subs);

%% Compute function value
if computeF
    Fvec = fh(xvals,mvals);
    if ~isempty(crng)
        Fvec(crng) = Fvec(crng) - fh(0,mvals(crng));
    end
    F = sum( weights .* Fvec );
end
if ~computeG
    return;
end

%% Compute sample y values
yvals = weights .* gh(xvals, mvals);
if ~isempty(crng)
    yvals(crng) = yvals(crng) - weights(crng) .* gh(0, mvals(crng));
end

%% Compute function and gradient
G = cell(d,1);
nsamples = size(subs,1);
for k=1:d
    % The row of each element is the row index to accumulate in the
    % gradient. The column indices are corresponding samples. They are
    % in order because they match the vector of samples to be
    % multiplied on the right.    
    S = sparse(subs(:,k), (1:nsamples)', yvals, sz(k), nsamples, nsamples);    
    G{k} = S * Zexp{k};
end

% Convert to single vector
if vectorG
    G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false));
end

%% If not computing F, set F (the 1st return arugment) to be the gradient
if ~computeF
    F = G;
end
