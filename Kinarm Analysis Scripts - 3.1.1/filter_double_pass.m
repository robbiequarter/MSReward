function dataOut = filter_double_pass(dataIn, method, varargin)
%FILTER_DOUBLE_PASS Double-pass filter all analog data in c3d data_structure.
%
%	DATA_OUT = FILTER_DOUBLE_PASS(DATA_IN, METHOD, 'fc', FC) filters
%	all floating-point channel data in the structure DATA_IN. The data are all
%	double-pass filtered using a 3rd order filter for zero-lag filtering, with a
%	3db cutoff of the 6th order final filter at FC Hz.  If not supplied, then
%	the sampling frequency is extracted from DATA_IN.  The METHOD parameter can
%	either be 'standard' or 'enhanced'.  The 'standard' method uses reflection
%	of data about the end-points to minimize starting and end transients
%	(similar to the filtfilt.m function in the Signal Processing toolbox).
%	The 'enhanced' method builds upon the standard method to provide a
%	better reflection, particularly when a data stream starts or ends at
%	the peak of a noise spike.
%
%	Note: a typical values for FC for human kinematic data is 10 Hz.
%   Note: channels with a name ending in _timestamp will not be filtered.
%
%	DATA_OUT = FILTER_DOUBLE_PASS(DATA_IN, METHOD, 'fc', FC, 'fs', FS)
%	performs the same filtering, but uses sampling frequency FS. 
%
%	DATA_OUT = FILTER_DOUBLE_PASS(DATA_IN, METHOD, 'coeff', B, A)
%	implements a double-pass, zero-lag custom filter defined by filter
%	coeffecients B and A.  The filter defined by B and A will be applied
%	twice (once forwards and once backwards) for double-pass filter that is
%	twice the order of the filter defined by B and A and with a 3dB cutoff
%	at the frequency at which there is 1.5 db attenuation for the filter
%	defined by B and A.  
% 
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_friction(data)
%     OR
%   dataNew.c3d = KINARM_add_friction(data(ii).c3d)
%
%   Copyright 2009-2021 BKIN Technologies Ltd

dataOut = dataIn;

if isempty(dataIn)
	return
end

if ~exist('method', 'var') || isempty(method) || ~ischar(method) || isempty(strmatch(method, {'standard', 'enhanced'}, 'exact'))
	error(['---> No method was specified, or was specified improperly. Must be ''standard'' or ''enhanced''.']);
end
filter.method = method;
%check varargin
filter.fcSpecified = false;					%default is that cutoff frequency is not specified
filter.fsSpecified = false;					%default is that sampling frequency is not specified
filter.coeffSpecified = false;				%default is that filter coefficents are not specified
x = 1;
while x <= length(varargin)
	if strncmpi(varargin{x}, 'fs', 2)
		x = x + 1;
		if length(varargin) >= x
			filter.fs = varargin{x};
		else
			error('---> Sampling frequency (fs) was not specified properly.  No filtering applied');
		end
		if ~isnumeric(filter.fs)
			error('---> Sampling frequency (fs) was not numeric.  No filtering applied');
		end
		filter.fsSpecified = true;
	elseif strncmpi(varargin{x}, 'fc', 2)
		x = x + 1;
		if length(varargin) >= x
			filter.fc = varargin{x};
			filter.fcSpecified = true;
		else
			error('---> Cutoff frequency (fc) was not specified properly.  No filtering applied');
		end
		if ~isnumeric(filter.fc)
			error('---> Cutoff frequency (fc) was not numeric.  No filtering applied');
		end
	elseif strncmpi(varargin{x}, 'coeff', 5)
		x = x + 1;
		filter.B = varargin{1};
		filter.A = varargin{2};
		%if either of the filter inputs are empty, do not filter the data
		if isempty(filter.B) || isempty(filter.A)
			error('---> One or more of the filter coefficients was empty.  No filtering applied');
		end
		if ~isnumeric(filter.B) || ~isnumeric(filter.A)
			error('---> One or more of the filter coefficients was not numeric.  No filtering applied');
		end
		filter.coeffSpecified = true;	    
	end
    x = x + 1;
end

if ~filter.coeffSpecified && ~filter.fcSpecified
	error('---> Cutoff frequency (fc) was not specified.  No filtering applied');
end

if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then filter each exam file, one at a time.
	for jj = 1:length(dataIn)
		dataOut(jj).c3d = FilterAllTrials(dataOut(jj).c3d, filter);
	end
else
	% legacy functionality, assuming that data_in = examFile(ii).c3d
	dataOut = FilterAllTrials(dataOut, filter);
end	    


function data = FilterAllTrials(data, filter)
% for each trial of data in dataIn
if ~filter.coeffSpecified
	if ~filter.fsSpecified
		filter.fs = data(1).ANALOG.RATE;
	end
	[filter.B, filter.A] = CreateFilterCoeffForDblpass(filter.fc, filter.fs);
	nReflect = round(filter.fs/filter.fc);
else
	nReflect = max(length(filter.A), length(filter.B));
end

%calculate the basis for the input delays (zi) for the filter
ziBasis = createZiBasis(filter.B, filter.A);
for ii = 1:length(data)
	
	%determine which fields of dataIn to filter.  Only filter those which are
	%numeric and NOT integers and do NOT contain _timestamp.
	names = fieldnames(data(ii));
   
	for jj = length(names):-1:1;
        isTimestamped = ~isempty(strfind(lower(names{jj}),'_timestamp'));
		if ~isnumeric(data(1).(names{jj})) || isinteger(data(1).(names{jj}))||isTimestamped
			names(jj) = [];
        end
	end

	for jj = 1:length(names)
		dataToFilter = data(ii).(names{jj});
		if ~isempty(dataToFilter)
			if strmatch(filter.method, 'standard', 'exact')
				data(ii).(names{jj}) = DoublePassFilterStandard(dataToFilter, nReflect, filter.B, filter.A, ziBasis);	
			else
				data(ii).(names{jj}) = DoublePassFilterEnhanced(dataToFilter, nReflect, filter.B, filter.A, ziBasis);	
			end
			
		end
	end
end
disp('Finished filtering data structure');


function	dataOut = DoublePassFilterStandard(dataIn, nReflect, B, A, ziBasis)
% The traditional approach to avoid end_condition problems is to reflect
% the data about the end-points, filter the longer data stream, and then
% cut off the extra data from the reflection.  In addition, use the first
% and last points of the longer data stream to set the initial conditions
% of the filter. 
%
% NOTE: in the future can probably reduce n_reflect for greater speed.  

% Choose the number of reflection points.  The duration of the reflection
% data was chosen as 1/fc, which is very conservative. 
len = length(dataIn);
if len < 2
% 	not enough data to filter
	dataOut = dataIn;
	return
end
nReflect = min(nReflect, len-1);

% Create reflection data and reflection points from original
% data
reflectStart = dataIn(2:nReflect + 1);
reflectEnd = dataIn(len - nReflect:len - 1);
rp1 = dataIn(1);
rp2 = dataIn(len);
% Add reflection data to original data
try
dataTemp = [2*rp1 - flipud(reflectStart); dataIn; 2*rp2 - flipud(reflectEnd)];
% Double-pass filter		
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
catch
	print('oops');
end
% Clip off extra data
dataOut = dataTemp(nReflect+1:len + nReflect);




function	dataOut = DoublePassFilterEnhanced(dataIn, nReflect, B, A, ziBasis)

len = length(dataIn);
if len < 2
% 	not enough data to filter
	dataOut = dataIn;
	return
end
nReflect = min(nReflect, len-1);

% STEP 1.  Filter the data as per standard operation
% reflect_start = data_in(2:n_reflect + 1);
% reflect_end = data_in(len - n_reflect:len - 1);
% rp1 = data_in(1);
% rp2 = data_in(len);
% % Add reflection data to original data
% data_temp = [2*rp1 - flipud(reflect_start); data_in; 2*rp2 - flipud(reflect_end)];
% % Double-pass filter		
% data_temp = flipud(filter(B, A, data_temp, zibasis * data_temp(1)));
% data_temp = flipud(filter(B, A, data_temp, zibasis * data_temp(1)));
% % Clip off extra data
% data_filt = data_temp(n_reflect+1:len + n_reflect);
dataFilt = DoublePassFilterStandard(dataIn, nReflect, B, A, ziBasis);

% STEP 2.  Calculate the residual (difference between original and
% filtered data).  Contains the noise and error.
residual = dataIn - dataFilt;						%Difference between original and filtered data.  Contains the noise and error

% STEP 3.  Filter the residual, but because the residual only contains
% noise and error, do NOT rotate about start/end points.  Noise should
% average to zero so a straight mirroring for the reflected data is
% appropriate to filter out the noise, but leave in the error
reflectStart = residual(2:nReflect + 1);
reflectEnd = residual(len - nReflect:len - 1);
% Add reflection data to original data.  
dataTemp = [flipud(reflectStart); residual; flipud(reflectEnd)];
% Double-pass filter		
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
% Clip off extra data
residualFilt = dataTemp(nReflect+1:len + nReflect);

% STEP 4.  Use filtered data for the data to reflect, use the filtered
% residual to choose point to reflect/rotate about and then filter the
% original data.
reflectStart = dataFilt(2:nReflect + 1);
reflectEnd = dataFilt(len - nReflect:len - 1);
rp1 = dataFilt(1) + residualFilt(1);
rp2 = dataFilt(len) + residualFilt(len);
% Add reflection data to original data
dataTemp = [2*rp1 - flipud(reflectStart); dataIn; 2*rp2 - flipud(reflectEnd)];
% Double-pass filter		
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
dataTemp = flipud(filter(B, A, dataTemp, ziBasis * dataTemp(1)));
% Clip off extra data
dataOut = dataTemp(nReflect+1:len + nReflect);


function	ziBasis = createZiBasis(B, A)
%calculate the basis for the input delays (zi), which are used to
%initialize the filter correctly.  The zibasis is then multiplied by the
%first element of the vector being filtered (this has the effect of
%assuming that the n data points previous to the data_in were equal to the
%first data point)  

%initialize zibasis 
filterOrder = length(B) - 1;
ziBasis = [];
diff = B - A;
for ii = 1:filterOrder
	ziBasis(ii,:) = [diff(ii+1:filterOrder+1) zeros(1,ii-1)];
end
ziBasis = ziBasis * ones(filterOrder,1);



function [B A] = CreateFilterCoeffForDblpass(fcFinal, fs)
% This function creates the coefficients for a 3rd order Butterworth filter
% which will be used subsequently as the basis of a double-pass (zero-lag) 6th
% order filter.  The input fc_final is the desired 3db cutoff frequency (Hz) of
% the 6th order filter.  fs is the sampling frequency (Hz).


% For a final 6th order filtering with a 3db cuttoff of fc, we need the 3rd
% order butterworth filter to have 1.5db cutoff at fc (because passing a
% given filter 2x over data_in will double the attenuation at all
% frequencies).  For 3rd order filters, the 3db cutoff is 1/0.864 above the
% 1.5 db attenuation (the correction factor is different for different order
% filters, e.g. it is 0.802 for 2nd order filters).

correction = 0.864;
fc = fcFinal/correction;
wc = fc * 2 * pi;		%convert from Hz to rad/s
T = 1/fs;				%

wcw = 2/T*tan(wc*T/2);

% The following coefficients were taken from Alarcon et al. (2000, J. Neurosci
% Meth. 104, 35-44)
A0 =   8 + 8*wcw*T + 4*wcw^2*T^2 +   wcw^3*T^3;
A1 = -24 - 8*wcw*T + 4*wcw^2*T^2 + 3*wcw^3*T^3;
A2 =  24 - 8*wcw*T - 4*wcw^2*T^2 + 3*wcw^3*T^3;
A3 =  -8 + 8*wcw*T - 4*wcw^2*T^2 +   wcw^3*T^3;

B = [1 3 3 1]*wcw^3*T^3/A0;
A = [A0 A1 A2 A3]/A0;

