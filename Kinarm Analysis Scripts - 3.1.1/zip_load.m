% EXAM_LOAD depricated function. Calls through to EXAM_LOAD
%   Copyright 2010-2021 BKIN Technologies Ltd
function c3dstruct = zip_load(varargin)
    if isempty(varargin)
        c3dstruct = exam_load();    
    else
        c3dstruct = exam_load(varargin{1});
    end
end