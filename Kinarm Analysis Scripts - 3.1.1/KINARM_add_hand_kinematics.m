function dataOut = KINARM_add_hand_kinematics(dataIn)
%KINARM_ADD_HAND_KINEMATICS Calculate hand velocity and accelerations.
%	DATA_OUT = KINARM_ADD_HAND_KINEMATICS(DATA_IN) calculates the hand
%	velocities, accelerations and commanded hand forces from the joint
%	velocities, accelerations and motor torques for the KINARM robot.
%	These data are added as new fields to the DATA_IN structure. 
%
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_friction(data)
%     OR
%   dataNew.c3d = KINARM_add_friction(data(ii).c3d)
%
%	The hand kinematics are calculated from joint kinematics rather than
%	differentiating the hand position because for some KINARM robots the
%	original joint kinematics are all calculated in real-time at 1.129 kHz
%	and then re-sampled to 1 kHz.  See the Dexterit-E User Guide for
%	more information.  Differentiating the hand position will be produce
%	significant noise.
%
%   The hand forces calculated here are the commanded hand forces as would
%   have been 'commanded' to the KINARM robot and are based on the torques
%   applied to the robot.  These forces do NOT include the effects robot
%   inertia, which is typically not compensated for when commanded a
%   particular force or torque.  The actual force applied at the hand can
%   be either estimated using the equations of motion or measured using a
%   Force/Torque sensor at the hand.
%
%	The new fields are in units of m/s, m/s^2 and N, and are in a global
%	coordinate system (as per Right_HandX, Left_HandY etc) and are:  
% 		.Right_HandXVel
% 		.Right_HandYVel
% 		.Right_HandXAcc
% 		.Right_HandYAcc
%		.Right_Hand_ForceCMD_X
%		.Right_Hand_ForceCMD_Y
% 		.Left_HandXVel
% 		.Left_HandYVel
% 		.Left_HandXAcc
% 		.Left_HandYAcc
%		.Left_Hand_ForceCMD_X
%		.Left_Hand_ForceCMD_X

%   Copyright 2009-2021 BKIN Technologies Ltd

%default output
dataOut = dataIn;

if isempty(dataIn)
	return;
end

if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then add hand kinematics to each exam file, one at a time.
	for jj = 1:length(dataIn)
		dataOut(jj).c3d = AddKinematicsToAllTrials(dataIn(jj).c3d, dataOut(jj).c3d);
    	dataOut(jj).c3d = ReorderFieldNames(dataIn(jj).c3d, dataOut(jj).c3d);
	end
else
	% legacy functionality, assuming that data_in = examFile(ii).c3d
	dataOut = AddKinematicsToAllTrials(dataIn, dataOut);
	dataOut = ReorderFieldNames(dataIn, dataOut);
end	

disp('Finished adding KINARM robot hand kinematics');

end

%%
function dataOut = AddKinematicsToAllTrials(dataIn, dataOut)
	for ii = 1:length(dataOut)
		for jj = 1:2
			if jj == 1;
				side = 'RIGHT';
				side2 = 'Right';
			else 
				side = 'LEFT';
				side2 = 'Left';
			end

			if ~isfield(dataOut(ii), [side '_KINARM']) || ~isfield(dataOut(ii).([side '_KINARM']), 'VERSION')
				continue
			end

			if isfield(dataIn(ii), [side2 '_HandX']) && (~isfield(dataIn(ii), [side2 '_KINARM']) || dataOut(ii).([side '_KINARM']).IS_PRESENT)
				% Check the version of the KINARM
				version = dataOut(ii).([side '_KINARM']).VERSION;
				if strncmp('KINARM_EP', version, 9)
					% KINARM End-Point robot
					L1 = dataOut(ii).([side '_KINARM']).L1_L;
					L2 = dataOut(ii).([side '_KINARM']).L2_L;
					L2PtrOffset = 0;
				elseif strncmp('KINARM_H', version, 8) || strncmp('KINARM_M', version, 8)
					% KINARM Exoskeleton robot
					L1 = dataOut(ii).CALIBRATION.([side '_L1']);
					L2 = dataOut(ii).CALIBRATION.([side '_L2']);
					L2PtrOffset = dataOut(ii).CALIBRATION.([side '_PTR_ANTERIOR']);
					if strcmp(side, 'LEFT')
						% L2PtrOffset is in global coordinates, not local, so change sign
						% as compared to the right hand.
						L2PtrOffset = -L2PtrOffset;
					end
				else
					% unidentified robot
					error(['unidentified ' side ' KINARM robot type']);
				end
				L1Ang = dataOut(ii).([side2 '_L1Ang']);
				L2Ang = dataOut(ii).([side2 '_L2Ang']);
				L1Vel = dataOut(ii).([side2 '_L1Vel']);
				L2Vel = dataOut(ii).([side2 '_L2Vel']);
				L1Acc = dataOut(ii).([side2 '_L1Acc']);
				L2Acc = dataOut(ii).([side2 '_L2Acc']);
				M1TorApp = dataIn(ii).([side2 '_M1TorCMD']);
				M2TorApp = dataIn(ii).([side2 '_M2TorCMD']);
				[hvx, hvy, hax, hay, Fx, Fy] = CalcHandKinematics(L1, L2, L2PtrOffset, L1Ang, L2Ang, L1Vel, L2Vel, L1Acc, L2Acc, M1TorApp, M2TorApp);
				dataOut(ii).([side2 '_HandXVel']) = hvx;
				dataOut(ii).([side2 '_HandYVel']) = hvy;
				dataOut(ii).([side2 '_HandXAcc']) = hax;
				dataOut(ii).([side2 '_HandYAcc']) = hay;
				dataOut(ii).([side2 '_Hand_ForceCMD_X']) = Fx;
				dataOut(ii).([side2 '_Hand_ForceCMD_Y']) = Fy;
			end
		end
	end
end
	

function dataOut = ReorderFieldNames(dataIn, dataOut)
	%re-order the fieldnames so that the hand velocity, acceleration and
	%commanded forces are with the hand position at the beginning of the field
	%list 
	origNames = fieldnames(dataIn);
	tempNames = fieldnames(dataOut);
	rightNames = {'Right_HandXVel'; 'Right_HandYVel'; 'Right_HandXAcc'; 'Right_HandYAcc'; 'Right_Hand_ForceCMD_X'; 'Right_Hand_ForceCMD_Y'};
	leftNames = {'Left_HandXVel'; 'Left_HandYVel'; 'Left_HandXAcc'; 'Left_HandYAcc'; 'Left_Hand_ForceCMD_X'; 'Left_Hand_ForceCMD_Y'};

	%check to see if any right-handed or left-handed fields were added to the
	%output data structure
	addedRightToOutput = false;
	addedLeftToOutput = false;
	for ii = 1:length(rightNames)
		if isempty( strmatch(rightNames{ii}, origNames, 'exact') ) && ~isempty( strmatch(rightNames{ii}, tempNames, 'exact') )
			addedRightToOutput = true;
		end
		if isempty( strmatch(leftNames{ii}, origNames, 'exact') ) && ~isempty( strmatch(leftNames{ii}, tempNames, 'exact') )
			addedLeftToOutput = true;
		end
	end

	if addedRightToOutput
		% remove all of the new fields from the original list
		for ii = 1:length(rightNames)
			index = strmatch(rightNames{ii}, origNames, 'exact');
			if ~isempty(index)
				origNames(index) = [];
			end
		end
		% place the new fields right after the HandY field
		index = strmatch('Right_HandY', origNames, 'exact');
		newNames = cat(1, origNames(1:index), rightNames, origNames(index+1:length(origNames)));
	else
		newNames = origNames;
	end

	if addedLeftToOutput
		% remove all of the new fields from the original list
		for ii = 1:length(leftNames)
			index = strmatch(leftNames{ii}, origNames, 'exact');
			if ~isempty(index)
				origNames(index) = [];
			end
		end
		% place the new fields right after the HandY field
		index = strmatch('Left_HandY', newNames, 'exact');
		newNames = cat(1, newNames(1:index), leftNames, newNames(index+1:length(newNames)));
	end
	dataOut = orderfields(dataOut, newNames);
end


function [hvx, hvy, hax, hay, Fx, Fy] = CalcHandKinematics(L1, L2, L2PtrOffset, L1Ang, L2Ang, L1Vel, L2Vel, L1Acc, L2Acc, T1, T2)
	% function which calculates hand velocity and acceleration from the angular
	% velocities and accelerations 

	sinL1 = sin(L1Ang);
	cosL1 = cos(L1Ang);
	sinL2 = sin(L2Ang);
	cosL2 = cos(L2Ang);
	sinL2ptr = cosL2;
	cosL2ptr = -sinL2;

	%hand velocities and accelerations
	hvx = -L1*sinL1.*L1Vel - L2*sinL2.*L2Vel - L2PtrOffset*sinL2ptr.*L2Vel;
	hvy = L1*cosL1.*L1Vel + L2*cosL2.*L2Vel + L2PtrOffset*cosL2ptr.*L2Vel;
	hax = -L1 * (cosL1.*L1Vel.^2 + sinL1.*L1Acc) - L2 * ( cosL2.*L2Vel.^2 + sinL2.*L2Acc) - L2PtrOffset * ( cosL2ptr.*L2Vel.^2 + sinL2ptr.*L2Acc);
	hay = L1 * (-sinL1.*L1Vel.^2 + cosL1.*L1Acc) + L2 * (-sinL2.*L2Vel.^2 + cosL2.*L2Acc) + L2PtrOffset * (-sinL2ptr.*L2Vel.^2 + cosL2ptr.*L2Acc);

	%hand forces
	A1 = -L1*sinL1;
	A2 = L1*cosL1;
	A3 = -(L2*sinL2+L2PtrOffset*sinL2ptr);
	A4 = (L2*cosL2+L2PtrOffset*cosL2ptr);
	%pre-allocate the memory for the _Hand_FX and _Hand_FY vectors
	%for enhanced speed
	Fx = L1Ang;
	Fy = L1Ang;
	for k = 1:length(L1Ang)
		F = [A1(k) A2(k); A3(k) A4(k)] \ [T1(k); T2(k)];
		Fx(k) = F(1);
		Fy(k) = F(2);
	end
end