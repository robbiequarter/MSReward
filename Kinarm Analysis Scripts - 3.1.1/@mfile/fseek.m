% FUNCTION pos = fseek(MF,OFFSET,ORIGIN)
%
% This operarates on "mfile" objects exactly like the "fseek" command
% file objects created with fopen.
%
%   STATUS = FSEEK(MF, OFFSET, ORIGIN) repositions the memory file
%    position inducator in the memory file MF. fseek sets the 
%    position indicator to the byte with the specified OFFSET relative to 
%    ORIGIN.
% 
%    MF is a memory file object
% 
%    OFFSET values are interpreted as follows:
%        >= 0    Move position indicator OFFSET bytes after ORIGIN.
%        < 0    Move position indicator OFFSET bytes before ORIGIN.
% 
%    ORIGIN values are interpreted as follows:
%        'bof' or -1   Beginning of file
%        'cof' or  0   Current position in file
%        'eof' or  1   End of file
% 
%    STATUS is 0 on success and -1 on failure.
%  
% Author: Steven Michael (smichael@ll.mit.edu)
% Date:   9/27/2007
%
