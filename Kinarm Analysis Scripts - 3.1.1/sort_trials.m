%SORT_TRIALS given a set of trials loaded from zip_load this will sort the
%   trials based on the criteria you specify.
%
%   sorted = sort_trials(zip_load()) Will sort the trials based on the
%   execution order.
%
%   sorted = sort_trials(zip_load(), [type], [method]). Type is one of:
%   'execution' - execution order sort
%   'tp' - sort by trial protocol number (and run order when tp's match).
%   'custom' - use the supplied method argument as the sorting method.
%   method - a pointer to a method with the signature sortMethod(c3d1,
%   c3d2). The method should return true when c3d1 > c3d2, false otherwise.

%   Copyright 2015-2021 BKIN Technologies Ltd

function examData = sort_trials(examData, varargin)
    % This method implements a simple bubble sort for the trials in exam
    % file data loaded using zip_load.
    
	if isempty(varargin) || strcmpi('execution', varargin{1})
        sortMethod = @sortByRunOrder;
    elseif strcmpi('tp', varargin{1})
        sortMethod = @sortByTP;
    elseif strcmpi('custom', varargin{1})
        sortMethod = varargin{2};
	end
    
	for file = 1:length(examData)
		n = length(examData(file).c3d);
    
		while (n > 0)
			% Iterate through each trial
			nnew = 0;
			for ii = 2:n
				% Swap elements in wrong order
				if sortMethod(examData(file).c3d(ii - 1), examData(file).c3d(ii))
					swap(ii, ii - 1);
					nnew = ii;
				end
			end
			n = nnew;
		end
	end
	
    function swap(i,j)
        val = examData(file).c3d(i);
        examData(file).c3d(i) = examData(file).c3d(j);
        examData(file).c3d(j) = val;
    end    
end

function ret = sortByRunOrder(c3d1, c3d2)
    ret = c3d1.TRIAL.TRIAL_NUM > c3d2.TRIAL.TRIAL_NUM;  
end

function ret = sortByTP(c3d1, c3d2)
    if c3d1.TRIAL.TP == c3d2.TRIAL.TP
        ret = sortByRunOrder(c3d1, c3d2);
    else
        ret = c3d1.TRIAL.TP > c3d2.TRIAL.TP;  
    end
end
