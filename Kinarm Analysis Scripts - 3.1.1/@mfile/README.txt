Description:

mfile (memory file) is a MATLAB class for reading binary data from memory.

The class is intended as a near drop-in replacement for fopen, fread, 
fseek, and ftell.  Parsing binary files in MALTAB using the standard 
fopen and fread can be time consuming.  A disk read operation is required
for each fread call.  the mfile class eliminates the need for multiple 
read operations by reading the binary file into memory all at once.  fread
commands using mfile scan the memory array holding the file contents rather
than accessing the disk.  For complicated file formats, the speed 
improvement can be significant.

Usage:

Make sure the directory containing the "@mfile" class directory is in the
MATLAB path.  Replace "fopen" command with "mfile" commands.  Operations
available on the "mfile" object are fseek,ftell, and fread.

the fread and fseek commands are MATLAB mex code compiled from C.  Binaries
are included for 32-bit Windows, 64-bit Linux, and 32-bit Linux. Compiling on
other platforms should be simple.  Type
>> mex fread.c
>> mex fseek.c
from the MATLAB command line.  Included binaries are compiled with Matlab
R2007b


Example:

  original code:
  
  % The line below is the original call one would use
  % for opening the file
  % fid=fopen('mydata.dat','r');
	
  % Instead, open a "memory file"
  fid=mfile('mydata.dat');

  % Read some integer values, 4 ints
  % at a time, skipping 32 bytes in between
  % and storing the data in int32 format
  val1=fread(fid,10,'4*int32=>int32',32);
	% Go 16 bytes from the beginning of the file
  fseek(fid,16,'bof');
  % Read some float32 values, 8 floats
  % at a time, skipping 32 bytes in between
  % and storing the data in double format
  val2=fread(fid,8,'single=>double');
  
Note:  This code has not been tested with every permutation possible
of data format conversions.  However, I have not had any difficulty with it
yet.  Please let me know if you find a bug.

Changes:

10/23/07  Update to handle '*<vartype>' shorthand and
          64-bit integer data types. Note: unsigned 64-bit
          integer does not work with Windows (I think it is
          a MATLAB compiler issue)
   

Author:  Steven Michael (smichael@ll.mit.edu)

  Copyright (c) 2007 Steven Michael (smichael@ll.mit.edu)
 
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
 
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
