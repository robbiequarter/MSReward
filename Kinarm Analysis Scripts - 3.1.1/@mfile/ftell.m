%FTELL Get memory file position indicator. 
%   POSITION = FTELL(MF) returns the location of the file position
%   indicator in the specified memory file.  Position is indicated in bytes
%   from the beginning of the file.  If -1 is returned, it indicates
%   that the query was unsuccessful. 
%
%   MF is a memory file obtained with mfile.
%
% Author: Steven Michael (smichael@ll.mit.edu)
% Date:   9/27/2007
%
function pos = ftell(mf)
pos = int32(mf.pos);
