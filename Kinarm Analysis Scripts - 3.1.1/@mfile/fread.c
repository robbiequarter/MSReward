/* 
 * Name:     fread.c
 * 
 * Author:   Steven Michael (smichael@ll.mit.edu)
 *
 * Date:     9/27/2007
 *
 * Description:
 *
 * This comples into a MATLAB function for use with the "mfile" class.
 * This code operates like the MATLAB "fread" function on a string of 
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

typedef unsigned short wchar;

static const char *ftypes[] =
	{
		"char",
		"int8",
		"uint8",
		"int16",
		"uint16",
		"int32",
		"uint32",
		"int64",
		"uint64",
		"float32",
		"single",
		"float64",
		"double",
		"unsigned char",
		"signed char",
		"integer*1",
		"integer*2",
		"integer*4",
		"integer*8",
		"real*4",
		"real*8"

	};
static mxClassID classIDs[] = 
	{
		mxCHAR_CLASS,
		mxINT8_CLASS,
		mxUINT8_CLASS,
		mxINT16_CLASS,
		mxUINT16_CLASS,
		mxINT32_CLASS,
		mxUINT32_CLASS,
		mxINT64_CLASS,
		mxUINT64_CLASS,
		mxSINGLE_CLASS,
		mxSINGLE_CLASS,
		mxDOUBLE_CLASS,
		mxDOUBLE_CLASS,
		mxUINT8_CLASS,
		mxINT8_CLASS,
		mxINT8_CLASS,
		mxINT16_CLASS,
		mxINT32_CLASS,
		mxINT64_CLASS,
		mxSINGLE_CLASS,
		mxDOUBLE_CLASS
	};

static int eSize(mxClassID id)
{
	switch (id) {
	case mxCHAR_CLASS:
	case mxINT8_CLASS:
	case mxUINT8_CLASS:
		return 1;
	case mxINT16_CLASS:
	case mxUINT16_CLASS:
		return 2;
	case mxINT32_CLASS:
	case mxUINT32_CLASS:
		return 4;
	case mxINT64_CLASS:
	case mxUINT64_CLASS:
		return 8;
	case mxSINGLE_CLASS:
		return 4;
	case mxDOUBLE_CLASS:
		return 8;
	default:
		return -1;
	}
	return 0;
}

static int parse_fstring(char *fstring,mxClassID *inFormat,
												 mxClassID *outFormat, int *toRead)
{
	char *s1,*s2;
	int defineOutputType = 0;
	int sameOutputType =  0;
	int nTypes = sizeof(classIDs)/sizeof(mxClassID);
	int i;
	*outFormat = mxDOUBLE_CLASS;

	s1 = fstring;
	if(sscanf(fstring,"%d*",toRead)==1) {
		s1 = strstr(fstring,"*")+1;
	}
	else
		toRead[0] = 1;

	s2 = strstr(fstring,"=>");
	if(s2) {
		defineOutputType = 1;
		*s2 = '\0';
	}
	
	if(s1[0]== '*') {
		if(defineOutputType== 1) {
			mexErrMsgTxt("Invalid String\n");
			return -1;
		}
		sameOutputType = 1;
		s1 = s1+1;
	}
	for(i=0;i<nTypes;i++) {
		if(!strcmp(ftypes[i],s1)) {
			inFormat[0] = classIDs[i];
			break;
		}
	}
	if(i==nTypes) {
		mexErrMsgTxt("Invalid input type\n");
		return -1;
	}
	
	if(defineOutputType) {
		s1 = s2+2;
		for(i=0;i<nTypes;i++) {
			if(!strcmp(ftypes[i],s1)) {
				outFormat[0] = classIDs[i];
				break;
			}
		}
		if(i==nTypes) {
			mexErrMsgTxt("Invalid output type\n");
			return -1;
		}
	}
	else if(sameOutputType )
		outFormat[0] = inFormat[0];
	else
		outFormat[0] = mxDOUBLE_CLASS;


	return 0;
	
} /* end of reading input types */
									



void mexFunction (int nlhs, mxArray *plhs[],
									int nrhs, const mxArray *prhs[])
{
	mwSize outDims[2] = {1,1};
	int nOut= 1;
	int mPos;
	unsigned char *mData;
	int mLength;
	char *fstring = (char *)0;
	mxClassID inFormat = mxDOUBLE_CLASS;
	mxClassID outFormat = mxDOUBLE_CLASS;
	int toRead = 1;
	int skip = 0;
	int bytesToRead = 0;
	int i,j;
	unsigned char *curPos;

	unsigned char *OutPtrU8 = (unsigned char *)0;
	char *OutPtrS8 = (char *) 0;
	unsigned short *OutPtrU16 = (unsigned short *)0;
	short *OutPtrS16 = (short *) 0;
	int *OutPtrS32 = (int *)0;
	unsigned int *OutPtrU32 = (unsigned int *)0;
	long long *OutPtrS64 = (long long *)0;
	unsigned long long *OutPtrU64 = (unsigned long long *)0;
	float *OutPtrSingle = (float *)0;
	double *OutPtrDouble = (double *)0;
	wchar *OutPtrWChar = (wchar *)0;
	int inSize;
	int nRead = 0;


	/* Check the input */
	if(nrhs < 3)
		mexErrMsgTxt("Invalid number of inputs");
	if(strcmp(mxGetClassName(prhs[0]),"mfile"))
		mexErrMsgTxt("First input must be mfile class");
	if(mxGetClassID(prhs[2])!=mxCHAR_CLASS)
		mexErrMsgTxt("Third argument must be a string");
	if(nrhs >3) 
		skip = (int) mxGetScalar(prhs[3]);

	/* Parse the input */
	mPos = (int) mxGetScalar(mxGetField(prhs[0],0,"pos"));
	mData = (unsigned char *)
		mxGetPr(mxGetField(prhs[0],0,"data"));
	mLength = mxGetNumberOfElements(mxGetField(prhs[0],0,"data"));

	/* Parse the format string */
	fstring = (char *)mxMalloc(mxGetNumberOfElements(prhs[2])+1);
	mxGetString(prhs[2],fstring,mxGetNumberOfElements(prhs[2])+1);
	if(parse_fstring(fstring,&inFormat,&outFormat,&toRead)) {
		mxFree(fstring);
		return;
	}
	mxFree(fstring);

	/* Get the size to output */
	if(mxGetClassID(prhs[1])==mxDOUBLE_CLASS) {
		if(mxGetNumberOfElements(prhs[1])==1) 
			outDims[1] = mxGetScalar(prhs[1]);
		else if(mxGetNumberOfElements(prhs[1])==2) {
			double *dptr = (double *)mxGetPr(prhs[1]);
			outDims[0] = (int) dptr[0];
			outDims[1] = (int) dptr[1];
		}
		else 
			mexErrMsgTxt("Can only read into arrays of at most 2 dimensions");
	}	
	else
		mexErrMsgTxt("Invalid size input\n");
	nOut = outDims[0]*outDims[1];

	/* Compute bytes to read */
	bytesToRead = outDims[0]*outDims[1]*eSize(inFormat);
	if(bytesToRead <= 0) {
		mexPrintf("bytes to read must be positive\n");
		return;
	}
	/*
	mexPrintf("%d %d %d\n",outDims[0],outDims[1],eSize(inFormat));
	mexPrintf("bytesToRead = %d\n",bytesToRead);
	*/
	/* Make sure we have enough elements to read the output */
	if(mLength-mPos < bytesToRead) {
		mexErrMsgTxt("Not enough elements to read in values\n");
		return;
	}
	
	/* Create an output matrix */
	plhs[0] = mxCreateNumericMatrix(outDims[0],outDims[1],outFormat,mxREAL);
	
	/* Try an easy case -- input & output format are the same */
	if(inFormat==outFormat && toRead == 1 && skip == 0 &&
		 inFormat!=mxCHAR_CLASS) {
		memcpy(mxGetPr(plhs[0]),mData+mPos,bytesToRead);
		*(mxGetPr(mxGetField(prhs[0],0,"pos"))) += bytesToRead;
		return;
	}
	
	/* Little more difficult -- input & output format are 
	 * the same, but with skips 
	 */
	else if(inFormat==outFormat && inFormat!=mxCHAR_CLASS) {
		unsigned char *outPtr = (unsigned char *) mxGetPr(plhs[0]);
		int readSize = toRead*eSize(inFormat);
		nRead = 0;
		curPos = mData+mPos;
		while(nRead <nOut) {		 
			memcpy(outPtr,curPos,readSize);
			outPtr += readSize;
			nRead += readSize;
			curPos += readSize+skip;
		}
		*(mxGetPr(mxGetField(prhs[0],0,"pos"))) += bytesToRead;
		return;
	}
	

	/* Most difficult case -- inputs and outputs
	 * are of different types
	 */


	curPos = mData+mPos;
	inSize = eSize(inFormat);
	switch(inFormat ) {		

		/* MATLAB char input */
	case mxCHAR_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32
            case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
                int tmp = (int) *((char *)curPos);
                OutPtrU64[i] = *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif           
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of MATLAB char input */


		/* unsigned char input */
	case mxUINT8_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((unsigned char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((unsigned char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((unsigned char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((unsigned char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32
            case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif       
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((unsigned char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of unsigned char input */

		/* char input */
	case mxINT8_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((char *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32
        case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((char *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		} 
		break;
		/* end of char input */

		/* unsigned short input */
	case mxUINT16_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((unsigned short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU8[i] = *((unsigned short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((unsigned short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((unsigned short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32
        case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((unsigned short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of unsigned short input */

		/* short input */
	case mxINT16_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS8[i] = *((short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((short *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((short *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		} 
		break;
		/* end of short input */


		/* unsigned int input */
	case mxUINT32_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((unsigned int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU8[i] = *((unsigned int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((unsigned int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((unsigned int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((unsigned int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of unsigned int input */

		/* int input */
	case mxINT32_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS8[i] = *((int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((int *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((int *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		} 
		break;
		/* end of int input */


		/* unsigned long long input */
	case mxUINT64_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((unsigned long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU8[i] = *((unsigned long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((unsigned long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((unsigned long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((unsigned long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of unsigned long long input */



		/* long long input */
	case mxINT64_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU8[i] = *((long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrS8[i] = (char) *((long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((long long *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((long long *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		}
		break;
		/* end of long long input */



		/* float32  input */
	case mxSINGLE_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((float *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((float *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS8[i] = *((float *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((float *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxDOUBLE_CLASS:
			OutPtrDouble = (double *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrDouble[i] = *((float *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		} 
		break;
		/* end of single input */


		/* float64  input */
	case mxDOUBLE_CLASS:			
		switch(outFormat) {
		case mxCHAR_CLASS:
			OutPtrWChar = (wchar *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrWChar[i] = (wchar) *((double *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT8_CLASS:
			OutPtrU8 = (unsigned char *)mxGetPr(plhs[0]);
			for(i=0;i<nOut;i++) {
				OutPtrU8[i] = (unsigned char) *((double *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0)) 
					curPos += skip;
			}
			break;
		case mxINT8_CLASS:
			OutPtrS8 = (char *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS8[i] = *((double *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT16_CLASS:
			OutPtrU16 = (unsigned short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU16[i] = *((double *)curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT16_CLASS:
			OutPtrS16 = (short *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS16[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxUINT32_CLASS:
			OutPtrU32 = (unsigned int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU32[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxINT32_CLASS:
			OutPtrS32 = (int *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS32[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#ifndef WIN32            
		case mxUINT64_CLASS:
			OutPtrU64 = (unsigned long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrU64[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
#endif            
		case mxINT64_CLASS:
			OutPtrS64 = (long long *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrS64[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		case mxSINGLE_CLASS:
			OutPtrSingle = (float *)mxGetPr(plhs[0]);
			for(i=0,j=0;i<nOut;i++,j++) {
				OutPtrSingle[i] = *((double *) curPos);
				nRead++;
				curPos += inSize;
				if(skip && (nRead%toRead==0))
					curPos += skip;
			}
			break;
		default:
			break;
		} 
		break;
		/* end of double input */

	default:
		break;
	} /* end of in format*/
	*(mxGetPr(mxGetField(prhs[0],0,"pos"))) += bytesToRead;
	return;
} /* end of mexFunction */
