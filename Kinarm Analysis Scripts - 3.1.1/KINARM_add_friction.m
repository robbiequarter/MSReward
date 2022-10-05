function dataOut = KINARM_add_friction(dataIn, varargin)
%	DATA_OUT = KINARM_ADD_FRICTION(DATA_IN)
%	Adds new fields to the DATA_IN structure containing estimates of
%	friction (including viscous damping). These estimates are based on
%	typical values for a KINARM robot, based on the KINARM robot type.
%
%	The combined effects of friction plus viscosity are calculated for each
%	motor/segment (M1/L1 and M2/L2) of the KINARM robot and stored as new
%	fields in the DATA_OUT structure.  The new fields are in units of Nm,
%	in a global coordinate system (as per Right_M1TorCMD etc) and are: 
%		.Right_M1TorFRC
%		.Right_M2TorFRC
%		.Left_M1TorFRC
%		.Left_M2TorFRC
%
%	Typically, this function should be used prior to filtering (i.e.
%	friction should be calculated on unfiltered data, and then subsequently
%	filtered, if filtering is desired).  
%
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_friction(data)
%     OR
%   dataNew.c3d = KINARM_add_friction(data(ii).c3d)
%
%	DATA_OUT = KINARM_ADD_FRICTION(DATA_IN, u, B)
%   If friction ('u', Nm) and viscous damping ('B', Nm / (rad/s) )
%   coefficients are supplied as input arguments to this function, then
%   those values are used instead of the defaults.    
%
%   The inputs u and B can be either scalars or vectors of length 2 or 4.
%   If they are scalar, then the values will be used for both joints on
%   both arms.  If they are vectors of length 2, then u(1) and B(1) will be
%   applied to motor 1 (M1) for each arm and u(2) and B(2) will be applied
%   to motor 2 (M2) for each arm (where M1 applies torque to L1 and M2
%   applied torque to L2).  If they are  vectors of length 4, then u(1)
%   through u(4) and B(1) through B(4)will be applied to Right_M1,
%   Right_M2, Left_M1 and Left_M2 respectively.   
%
%	u must be in units of Nm and B must be in units of
%	Nm/(rad/s).  Although every  robot has slightly different friction
%	coeffecients, typical values are:
%     KINARM Exoskeleton Classic - KINARM_ADD_FRICTION(DATA_IN, 0.12, 0.0009)
%     Human KINARM Exoskeleton   - KINARM_ADD_FRICTION(DATA_IN, 0.15, 0.0014)
%     KINARM End-Point           - KINARM_ADD_FRICTION(DATA_IN, 0.18, 0.0020)
%
%
%   Copyright 2009-2021 BKIN Technologies Ltd

dataOut = dataIn;

if isempty(dataIn)
	return
end

if isempty(varargin)
	useDefaultFriction = true;
else
	useDefaultFriction = false;
	if length(varargin) == 1
		error('WARNING: one of friction or viscosity was not specified.  Friction and viscosity cannot be added');
	else
		u = varargin{1};
		B = varargin{2};
		if ~(length(u)==1 || length(u)==2 || length(u)==4 )
			error('WARNING: length of friction input incorrect.  Friction and viscosity cannot be added');
		end

		if ~(length(B)==1 || length(B)==2 || length(B)==4 )
			error('WARNING: length of viscosity input incorrect.  Friction and viscosity cannot be added');
		end
		
		%expand u and/or B to be 1x4 if needed
		if length(u) == 1
			u = u * [1 1 1 1];
		elseif length (u) == 2
			u = reshape([u u],1,4);
		end

		if length(B) == 1
			B = B * [1 1 1 1];
		elseif length (B) == 2
			B = reshape([B B],1,4);
		end
	end
end


if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then add friction to each exam file, one at a time.
	for jj = 1:length(dataIn)
		% Determine the appropriate friction coefficients for each exam
		% file, based on the KINARM type in that exam file 
		if useDefaultFriction
			[u, B] = GetFrictionCoeff(dataIn(jj).c3d);
		end
		dataOut(jj).c3d = AddFrictionToAllTrials(dataOut(jj).c3d, u, B);
        dataOut(jj).c3d = ReorderFieldNames(dataIn(jj).c3d, dataOut(jj).c3d);
	end
else
	% legacy functionality, assuming that data_in = examFile(ii).c3d
	if useDefaultFriction
		[u, B] = GetFrictionCoeff(dataIn);
	end
	dataOut = AddFrictionToAllTrials(dataOut, u, B);
	dataOut = ReorderFieldNames(dataIn, dataOut);
end	

disp('Finished adding KINARM robot friction to all files.');

end

%%
function dataOut = ReorderFieldNames(dataIn, dataOut)
	%re-order the fieldnames so that the friction forces are with the motor torques
	originalNames = fieldnames(dataIn);
	rightHandNames = {'Right_M1TorFRC'; 'Right_M2TorFRC'};
	leftHandNames = {'Left_M1TorFRC'; 'Left_M2TorFRC'};
	%Before re-arranging them, check to see if they existed in the original
	%data_in structure, in which case do NOT re-arrange
	if ~isempty(strmatch('Right_L1Vel', originalNames, 'exact')) && isempty(strmatch(rightHandNames{1}, originalNames, 'exact'))
		index = max(strmatch('Right_M2TorCMD', originalNames));
		newNames = cat(1, originalNames(1:index), rightHandNames, originalNames(index+1:length(originalNames)));
	else
		newNames = originalNames;
	end
	if ~isempty(strmatch('Left_L1Vel', originalNames, 'exact')) && isempty(strmatch(leftHandNames{1}, originalNames, 'exact'))
		index = max(strmatch('Left_M2TorCMD', newNames));
		newNames = cat(1, newNames(1:index), leftHandNames, newNames(index+1:length(newNames)));
	end

	dataOut = orderfields(dataOut, newNames);

end

function [u, B] = GetFrictionCoeff(data)
	[uRight, BRight] = GetFrictionCoeffOneArm(data, 'RIGHT_KINARM');
	[uLeft, BLeft] = GetFrictionCoeffOneArm(data, 'LEFT_KINARM');
	u = [uRight uLeft];
	B = [BRight BLeft];
end

function [u, B] = GetFrictionCoeffOneArm(data, kinarm)
	% get the KINARM type from the data file and use that to provide a
	% friction estimate
	if isfield(data, kinarm) && isfield(data(1).(kinarm), 'IS_PRESENT') && data(1).(kinarm).IS_PRESENT == 1
		kinarmType = data(1).(kinarm).VERSION;
		if ~isempty(regexpi(kinarmType, '_H_R'))
			% Human Exoskeleton Classic
			u = [0.12 0.12];
			B = [0.0009 0.0009];
		elseif ~isempty(regexpi(kinarmType, '_HUTS_R'))
			% Human Exoskeleton (UTS version)
			u = [0.15 0.15];
			B = [0.0014 0.0014];
		elseif ~isempty(regexpi(kinarmType, '_M_R'))
			% NHP Exoskeleton Classic
			u = [0.08 0.04];
			B = [0.0006 0.0002];
		elseif ~isempty(regexpi(kinarmType, '_M_UTS_R'))
			% NHP Exoskeleton (UTS version)
			u = [0.08 0.06];
			B = [0.0006 0.0003];
		elseif ~isempty(regexpi(kinarmType, '_EP_R'))
			% EP
			u = [0.18 0.18];
			B = [0.0020 0.0020];
		else
			error('KINARM type in data file is not recognized');
		end
	else
		u = [0 0];
		B = [0 0];
	end
end

function dataOut = AddFrictionToAllTrials(dataIn, u, B)
	dataOut = dataIn;
	for ii = 1:length(dataIn)
		%Right hand first.  Check to see if there is right hand data.
		if isfield(dataIn(ii), 'Right_L1Vel');
			% tanh(100*v) is a reasonable model of friction
			v = dataIn(ii).Right_L1Vel;
			dataOut(ii).Right_M1TorFRC = -u(1)*tanh(100*v) - B(1)*v;
			v = dataIn(ii).Right_L2Vel;
			dataOut(ii).Right_M2TorFRC = -u(2)*tanh(100*v) - B(2)*v;
		end
		if isfield(dataIn(ii), 'Left_L1Vel');
			% tanh(100*v) is a reasonable model of friction
			v = dataIn(ii).Left_L1Vel;
			dataOut(ii).Left_M1TorFRC = -u(3)*tanh(100*v) - B(3)*v;
			v = dataIn(ii).Left_L2Vel;
			dataOut(ii).Left_M2TorFRC = -u(4)*tanh(100*v) - B(4)*v;
		end
	end
end