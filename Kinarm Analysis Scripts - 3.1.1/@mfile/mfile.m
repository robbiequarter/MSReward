% FUNCTION mf = mfile(filename)
%
%
% Description:
%
%                     This creates a "memory file" object.  The object
%                     loads a file (or a data stream) into memory.  The
%                     memory is accessed with "fseek","ftell", and "fread"
%                     exactly as a file object created with "fopen".  In
%                     many cases, this should provide faster parsing of 
%                     binary input files than the standard 
%                     fopen / fread commands.
%
% Inputs:
%
%    filename    :    The name of a file to read into memory.  The 
%                     contents of the file must be able to fit into 
%                     available memory. 
%
%    data        :    An optional input used in place of the filename. 
%                     this must be an array of type 'uint8' that will 
%                     take the place of the data read by the file
% 
% Outputs:
% 
%    mf           :   The "mfile" object
%
%
%
% Author: Steven Michael (smichael@ll.mit.edu)
% Date:   9/27/2007
% 
function mf = mfile(varargin)
  
  mf.filename = [];
  mf.data = [];
  mf.pos = 0;  
  mf = class(mf,'mfile');
  if nargin==0, return;end;

  if isstruct(varargin{1});
    mf = class(varargin{1},mfile);
  elseif ischar(varargin{1});
    mf.filename = varargin{1};
    fid=fopen(mf.filename);
    if fid<1
      error('Invalid file');
    end
    mf.data = fread(fid,inf,'uint8=>uint8');
  elseif isa(varargin{1},'uint8')
    mf.data = varargin{1};
  end
  
      
  
  
