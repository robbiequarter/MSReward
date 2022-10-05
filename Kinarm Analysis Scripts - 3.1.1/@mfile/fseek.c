/* 
 * Name:     fseek.c
 * 
 * Author:   Steven Michael (smichael@ll.mit.edu)
 *
 * Date:     9/27/2007
 *
 * Description:
 *
 * This comples into a MATLAB function for use with the "mfile" class.
 * This code operates like the MATLAB "fseek" function on a string of 
 * "uint8" loaded into memory.  The code does not currently handle
 * endian switching.
 *  
 * Copyright (c) 2007 Steven Michael (smichael@ll.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <mex.h>
#include <string.h>

typedef enum {
	MFILE_BOF = -1,
	MFILE_COF = 0,
	MFILE_EOF = 1
} MFilePosRef;

static int get_posref(const mxArray *mx,MFilePosRef *posRef)
{
	if(mxIsNumeric(mx)) {
		int val = (int) mxGetScalar(mx);
		if(val<-1 || val>1) {
			mexPrintf("Position reference must be -1, 0, or 1\n");
			return -1;
		}
		*posRef = (MFilePosRef) val;
	}
	else if(mxIsChar(mx)) {
		char cval[4];
		if(mxGetNumberOfElements(mx)!=3) {
			mexPrintf("Invalid position reference string\n");
			return -1;
		}
		mxGetString(mx,cval,4);
		if(!strcmp(cval,"bof"))
			*posRef = MFILE_BOF;
		else if(!strcmp(cval,"cof"))
			*posRef = MFILE_COF;
		else if(!strcmp(cval,"eof")) 
			*posRef = MFILE_EOF;
		else {
			mexPrintf("Invalid position reference string\n");
			return -1;
		}
	}
	else {
		mexPrintf("Invalid position reference variable\n");
		return -1;
	}

	return 0;
}

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	MFilePosRef posRef = MFILE_BOF;
	long offset;
	long curPos;
	long mLength;
	long newPos;
	double dNewPos;

	/* Create a default output */
	plhs[0] = mxCreateDoubleScalar(-1.0);

	/* Verify valid input */
	if(nrhs < 2){ 
		mexPrintf("Invalid number of inputs.\n");
		return;
	}
	if(strcmp(mxGetClassName(prhs[0]),"mfile")) {
		mexPrintf("First input must be mfile class");
		return;
	}
	if(!mxIsNumeric(prhs[1])) {
		mexPrintf("Second argument must be numeric.\n");
		return;
	}

	/* Get the offset */
	offset = (long) mxGetScalar(prhs[1]);
	mLength=  (long) mxGetNumberOfElements(mxGetField(prhs[0],0,"data"));
	curPos = (long) mxGetScalar(mxGetField(prhs[0],0,"pos"));
	
	
	/* Get the position reference */
	if(nrhs > 2)
		if(get_posref(prhs[2],&posRef))
			return;
	
	/* Get the new position of the file pointer */
	switch (posRef)  {
	case MFILE_BOF:
		newPos = offset;
		break;
	case MFILE_EOF:
		newPos = mLength+offset-1;
		break;
	case MFILE_COF:
		newPos= curPos+offset;
		break;
	}
	if(newPos<0 || newPos>=mLength) {
		mexPrintf("Invalid file pointer position\n");
		return;
	}
	
	/* Set the position pointer to the appropriate value */
	dNewPos = (double)newPos;
	((double *)(mxGetPr(mxGetField(prhs[0],0,"pos"))))[0] = dNewPos;
	plhs[0] = mxCreateDoubleScalar(0.0);
	return;
}
