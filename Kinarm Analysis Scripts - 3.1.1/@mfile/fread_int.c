#include <mex.h>
#include <string.h>

typedef unsigned short wchar;
void mexFunction (int nlhs, mxArray *plhs[],
									int nrhs, const mxArray *prhs[])
{
	mwSize outDims[1] = {1};
	int mPos;
	unsigned char *mData;
	int *mDataAsInt;
	int mLength, nDataSize, i, swap;
//     unsigned char *start_of_pr;
    double *start_of_pr;
    size_t bytes_to_copy;

	/* Check the input */
	if(nrhs < 3)
		mexErrMsgTxt("Invalid number of inputs");
	if(strcmp(mxGetClassName(prhs[0]),"mfile"))
		mexErrMsgTxt("First input must be mfile class");

    outDims[0] = mxGetScalar(prhs[1]);
    swap = mxGetScalar(prhs[2]);
    
	/* Parse the input */
	mPos = (int) mxGetScalar(mxGetField(prhs[0],0,"pos"));
	mData = (unsigned char *)mxGetPr(mxGetField(prhs[0],0,"data"));
	mLength = mxGetNumberOfElements(mxGetField(prhs[0],0,"data"));
    
    bytes_to_copy = outDims[0] * 4;
    
    if (bytes_to_copy + mPos > mLength)
    {
		mexErrMsgTxt("Requesting more data than is available.");
        return;
    }
        
//     plhs[0] = mxCreateNumericArray(1,outDims,mxINT32_CLASS,mxREAL);
//     start_of_pr = (unsigned char *)mxGetData(plhs[0]);    
//     memcpy(start_of_pr,&mData[mPos],bytes_to_copy);

    plhs[0] = mxCreateNumericArray(1,outDims,mxDOUBLE_CLASS,mxREAL);
    start_of_pr = (double *)mxGetData(plhs[0]);    
    
    if (swap)
    {
        int dest, src;
        src = mPos;
        for (i = 0; i < outDims[0]; i++)
        {
            dest = (int)mData[src] << 24 |
                    (int)mData[src+1] << 16 |
                    (int)mData[src+2] << 8 |
                    (int)mData[src+3];
            start_of_pr[i] = (double)dest;
            src +=4;

        }
    }
    else
    {
        mDataAsInt = (int*)&mData[mPos];
        for (i =0; i < outDims[0]; ++i)
        {
            start_of_pr[i] = (double)mDataAsInt[i];
        }
    }
    
    *(mxGetPr(mxGetField(prhs[0],0,"pos"))) += bytes_to_copy;
}