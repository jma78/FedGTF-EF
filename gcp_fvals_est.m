function F = gcp_fvals_est(M, fh, subs, xvals, weights, LambdaCheck, crng)
%% Parse inputs
if nargin < 11
    
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

if LambdaCheck && ~all(M.lambda == 1)
    warning('Fixing M to have all ones for lambda');
    M = normalize(M,1);
end

%% Compute model values and exploded Zk matrices
[mvals, Zexp] = gcp_fg_est_helper(M.u, subs);

%% Compute function value
Fvec = fh(xvals,mvals);
if ~isempty(crng)
    Fvec(crng) = Fvec(crng) - fh(0,mvals(crng));
end
F = sum( weights .* Fvec );
