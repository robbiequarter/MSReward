%FREAD  Read binary data from file.
%
% data=fread(mf,size,format,skip)
%
% Description:
%
%  This operates on a memory file object in a manner similar to to
%  "fread" for file objects
%
%  There are some differences enumarated below:
%
%  1. fread currently does not handle endian switching. 
%
%  2. Size and format must be specified -- these are not optional.
%
%  3. inf cannot be used for the size;
%
% Author: Steven Michael (smichael@ll.mit.edu)
% Date:   9/27/2007
%