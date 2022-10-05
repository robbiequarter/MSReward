
function dataOut = KINARM_add_torques(dataIn)
%KINARM_ADD_TORQUES Estimate intramuscular and applied torques for KINARM 
%	Exoskeleton robot and applied forces at the hand for KINARM EP robots.
% 
%	The intramuscular torques are estimates of what torques the subject's
%	muscles must have produced based on the motion of the joints, the commanded
%	torques, friction and both robot and subject inertia.
%
%	The applied torques and applied forces are estimates of what torques/forces
%	the robot must have applied to the subject based on the motion of the
%	joints, the commanded torques, friction and robot.
%
%	DATA_OUT = KINARM_ADD_TORQUES(DATA_IN).  Calculations are based on the
%	kinematics in DATA_IN, the inertia of the KINARM robot, the inertia of
%	KINARM arm troughs (or anything else attached the KINARM robot), and
%	the inertia of the subject.  This function also calculates the torques
%	applied to the subject at the joints and end-point (i.e. hand).  
%
%	Typically, this function should be used after to filtering (i.e. torque
%	estimates should be calculated on filtered data). The rationale for this
%	approach is because the acceleration data used to estimate the torques
%	and/or the forces tends to be noisy
%
%	The new fields are:
%		[Right|Left]_Hand_[FX|FY]		(forces applied to subject's hand)
%		[Right|Left]_[ELB|SHO]TorIM		(intramuscular torques at joints)
%		[Right|Left]_[ELB|SHO]TorAPP	(torques applied to subject's joints)
%
%	Both [ELB|SHO]TorIM and [ELB|SHO]TorAPP are in a local coordinate frame, with
%	flexion positive (in contrast to the commanded torques in the data structure
%	which are in a global coordinate frame). The Hand_[FX|FY] data are in the
%	global coordinate frame.   
%
%	If subject inertial parameters are not present in the DATA_IN
%	structure, then intramuscular torques will not be calculated (only the
%	applied torques will be calculated).  Subject inertial parameters are
%	usually added to the DATA_IN structure by calling
%	KINARM_ADD_SUBJECT_INERTIA.
%
%	If arm troughs, handles or other masses were added the KINARM robot
%	during data collection, then the inertial properties of those extra
%	links must be added to the DATA_IN structure prior to calling this
%	function by calling KINARM_ADD_TROUGH_INERTIA.
%
%	Note: if the input DATA_IN structure does not have motor friction
%	estimates in it (e.g. DATA_IN(ii).Right_M1TorFRC), torques and forces
%	are still calculated, but a warning is provided.  Customers can add
%	their own friction values using KINARM_ADD_FRICTION.m 
%
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_friction(data)
%     OR
%   dataNew.c3d = KINARM_add_friction(data(ii).c3d)
%
%	The equations of motion for the KINARM robot were derived and provided by:
%	Dr. Gregory W Ojakangas
%	Associate professor of physics
%	Drury University
%	900 N Benton
%	Springfield, MO  65802
%   
%	and were published in the following article:
%	Fagg et al. (2009). Kinetic Trajectory Decoding Using
%   Motor Cortical Ensembles. IEEE Trans Neur Sys & Rehab Eng. 17(5):
%   487-496.
%   NOTE: there is a small difference in coordinate frames between the
%   equations published as part of Fagg et al. (2009) and those used here.
%   In particular, theta2 is defined differently, which is why the
%   equations here look different than those in Fagg et al. We defined
%   theta2 as the angle of link 2 in global coordinates, whereas Fagg et
%   al. define theta2 as the angle between link 2 and link 1.


%   Copyright 2009-2021 BKIN Technologies Ltd

dataOut = dataIn;

if isempty(dataIn)
	return
end

if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then add torques to each exam file, one at a time.
    for jj = 1:length(dataIn)
		fprintf('Calculating torques ');
		dataOut(jj).c3d = AddTorqueToAllTrials(dataIn(jj).c3d, dataOut(jj).c3d, dataOut(jj).filename{:} );
		disp( ['Finished adding KINARM robot applied and intramuscular torques to ' dataOut(jj).filename{:}] );
     	dataOut(jj).c3d = ReorderFieldNames(dataIn(jj).c3d, dataOut(jj).c3d);
    end
else
	% legacy functionality, assuming that data_in = examFile(ii).c3d
	dataOut = AddTorqueToAllTrials(dataIn, dataOut, 'legacy');
	dataOut = ReorderFieldNames(dataIn, dataOut);
	disp('Finished adding KINARM robot applied and intramuscular torques');
end	


end

%%
function dataOut = AddTorqueToAllTrials(dataIn, dataOut, examFilename)
	flags.RightIMTorqueCalc = true;
	flags.LeftIMTorqueCalc = true;
	flags.RightTroughInertia = true;
	flags.LeftTroughInertia = true;
	flags.RightFriction = true;
	flags.LeftFriction = true;
	for ii = 1:length(dataIn)
		for jj = 1:2
			if jj == 1;
				side = 'RIGHT';
				side2 = 'Right';
			else 
				side = 'LEFT';
				side2 = 'Left';
			end
			if isfield(dataIn(ii), [side2 '_HandX']) && ((isfield(dataIn(ii).([side '_KINARM']), 'IS_PRESENT') && dataOut(ii).([side '_KINARM']).IS_PRESENT) || (isfield(dataIn(ii).([side '_KINARM']), 'L1_I')))
				version = dataIn(ii).([side '_KINARM']).VERSION;
				KINARM_inertia = dataIn(ii).([side '_KINARM']);
				if isfield(dataIn(ii), [side '_KINARM_TROUGHS'])
					% Add inertia from auxillary objects (e.g. arm troughs)
					KINARM_inertia = KINARM_combine_inertias(KINARM_inertia, dataIn(ii).([side '_KINARM_TROUGHS']));
				else
					if strcmp(examFilename, 'legacy')
						disp(['WARNING - no inertias found for ' side ' KINARM arm troughs parts for trial ' dataIn(ii).FILE_NAME '.']);
					else
						flags.([side2 'TroughInertia']) = false;
					end
				end
				% ***********************************
				%calculate total torques applied to robot's arm segments (global coordinates)
				[T1, T2] = CalcTorques(KINARM_inertia, dataOut(ii), side, side2);

				%include the effects of friction to the Torques applied by the
				%motors if the frictions exist
				if isfield(dataIn(ii), [side2 '_M1TorFRC']) && isfield(dataIn(ii), [side2 '_M2TorFRC'])
					M1TorApp = dataIn(ii).([side2 '_M1TorCMD']) + dataIn(ii).([side2 '_M1TorFRC']);
					M2TorApp = dataIn(ii).([side2 '_M2TorCMD']) + dataIn(ii).([side2 '_M2TorFRC']);
				else
					if strcmp(examFilename, 'legacy')
						disp(['WARNING - no motor friction estimates found for trial ' dataIn(ii).FILE_NAME '.']);
					else
						flags.([side2 'Friction']) = false;
					end
					M1TorApp = dataIn(ii).([side2 '_M1TorCMD']);
					M2TorApp = dataIn(ii).([side2 '_M2TorCMD']);
				end

				%subtract the torques applied by the motors to calculate the
				%torque applied to the robot segments by the subject.
				T1 = T1 - M1TorApp;
				T2 = T2 - M2TorApp;
				
				%Calculate the torque applied by the robot to the
				%subject.
				T1 = -T1;
				T2 = -T2;
				
				%convert to local joint coordinates
				[TELB, TSHO] = ConvertTorques(T1, T2, side2);
				dataOut(ii).([side2 '_ELBTorAPP']) = TELB;
				dataOut(ii).([side2 '_SHOTorAPP']) = TSHO;

				% ***********************************
				% Calculate applied endpoint forces _Hand_FX and _Hand_FY
				if strncmp('KINARM_EP', version, 9)
					L1 = dataIn(ii).([side '_KINARM']).L1_L;
					L2 = dataIn(ii).([side '_KINARM']).L2_L;
					L2_ptr = 0;
				else
					% assume that it is an Exoskeleton robot.
					L1 = dataIn(ii).CALIBRATION.([side '_L1']);
					L2 = dataIn(ii).CALIBRATION.([side '_L2']);
					L2_ptr = dataIn(ii).CALIBRATION.([side '_PTR_ANTERIOR']);
				end
				L1Ang = dataIn(ii).([side2 '_L1Ang']);
				L2Ang = dataIn(ii).([side2 '_L2Ang']);
				if strmatch(side2, 'Right')
					L2ptr_Ang = L2Ang + pi;
				else
					L2ptr_Ang = L2Ang - pi;
				end

				A1 = -L1*sin(L1Ang);
				A2 = L1*cos(L1Ang);
				A3 = -(L2*sin(L2Ang)+L2_ptr*sin(L2ptr_Ang));
				A4 = (L2*cos(L2Ang)+L2_ptr*cos(L2ptr_Ang));
				%pre-allocate the memory for the _Hand_FX and _Hand_FY vectors
				%for enhanced speed
				dataOut(ii).([side2 '_Hand_FX']) = L1Ang;
				dataOut(ii).([side2 '_Hand_FY']) = L1Ang;
				for k = 1:length(L1Ang)
					F = [A1(k) A2(k); A3(k) A4(k)] \ [T1(k); T2(k)];
					dataOut(ii).([side2 '_Hand_FX'])(k) = F(1);
					dataOut(ii).([side2 '_Hand_FY'])(k) = F(2);
				end

				% ***********************************
				% calculate intramuscular torques _ELBTorIM and _SHOTorIM
				% But only if subject inertia exists.
				if isfield(dataIn(ii), [side '_ARM'])
					KINARM_subj_inertia = KINARM_combine_inertias(KINARM_inertia, dataIn(ii).([side '_ARM']));

					[T1, T2] = CalcTorques(KINARM_subj_inertia, dataOut(ii), side, side2);
					% subtract the torques applied by the motors
					T1 = T1 - M1TorApp;
					T2 = T2 - M2TorApp;
					% convert to local joint coordinates
					[TELB, TSHO] = ConvertTorques(T1, T2, side2);
					dataOut(ii).([side2 '_ELBTorIM']) = TELB;
					dataOut(ii).([side2 '_SHOTorIM']) = TSHO;
				else
					if strcmp(examFilename, 'legacy')
						disp(['WARNING - no subject inertial parameters found for trial ' dataIn(ii).FILE_NAME '.  Intramuscular torques not calculated']);
					else
						flags.([side2 'IMTorqueCalc']) = false;
					end
					dataOut(ii).([side2 '_ELBTorIM']) = [];
					dataOut(ii).([side2 '_SHOTorIM']) = [];
				end
			end
		end
		if ~strcmp(examFilename, 'legacy')
			fprintf('.');
		end
	end
	if ~strcmp(examFilename, 'legacy')
		fprintf('\n');
		if ~flags.RightFriction
			disp(['   WARNING - no motor friction estimates found for right arm in ' examFilename ]);
		end
		if ~flags.LeftFriction
			disp(['   WARNING - no motor friction estimates found for left arm in ' examFilename ]);
		end
		if ~flags.RightTroughInertia
			disp(['   WARNING - no inertias found for right KINARM arm troughs parts in ' examFilename ]);
		end
		if ~flags.LeftTroughInertia
			disp(['   WARNING - no inertias found for left KINARM arm troughs parts in ' examFilename ]);
		end
		if ~flags.RightIMTorqueCalc
			disp(['   WARNING - no subject inertial parameters found for right arm in ' examFilename '.  Intramuscular torques not calculated.']);
		end
		if ~flags.LeftIMTorqueCalc
			disp(['   WARNING - no subject inertial parameters found for left arm in ' examFilename '.  Intramuscular torques not calculated.']);
		end
	end
end


function dataOut = ReorderFieldNames(dataIn, dataOut)
	% re-order the fieldnames so that the hand forces are with the hand
	% kinematics and the joint torques are with motor torques
	origNames = fieldnames(dataIn);
	tempNames = fieldnames(dataOut);
	rightHandNames = {'Right_Hand_FX'; 'Right_Hand_FY'};
	leftHandNames = {'Left_Hand_FX'; 'Left_Hand_FY'};
	rightJointNames = {'Right_ELBTorIM'; 'Right_SHOTorIM'; 'Right_ELBTorAPP'; 'Right_SHOTorAPP'};
	leftJointNames = {'Left_ELBTorIM'; 'Left_SHOTorIM'; 'Left_ELBTorAPP'; 'Left_SHOTorAPP'};
	rightNames = [rightHandNames; rightHandNames];
	leftNames = [leftHandNames; leftJointNames];

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
		% place the new fields right after the HandY field and the M2Tor field
		index = max(strmatch('Right_HandY', origNames));		%Find last field beginning with 'Right_HandY'
		newNames = cat(1, origNames(1:index), rightHandNames, origNames(index+1:length(origNames)));
		index = max(strmatch('Right_M2Tor', newNames));
		newNames = cat(1, newNames(1:index), rightJointNames, newNames(index+1:length(newNames)));
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
		% place the new fields right after the HandY field and the M2Tor field
		index = max(strmatch('Left_HandY', newNames));
		newNames = cat(1, newNames(1:index), leftHandNames, newNames(index+1:length(newNames)));
		index = max(strmatch('Left_M2Tor', newNames));
		newNames = cat(1, newNames(1:index), leftJointNames, newNames(index+1:length(newNames)));
	end
	dataOut = orderfields(dataOut, newNames);
end



function [TELB, TSHO] = ConvertTorques(T1, T2, side2)
	% Convert torques from global to local coordinates
	TelbowGlobal = T2;
	Tshoulder_global = T1 + TelbowGlobal;
	if strmatch(side2, 'Right', 'exact')
		TELB = TelbowGlobal;
		TSHO = Tshoulder_global;
	else
		TELB = -TelbowGlobal;
		TSHO = -Tshoulder_global;
	end

end


function [T1, T2] = CalcTorques(inertia, data_in, side, side2)
	% This function calculates the total torques applied to each
	% segment (i.e. motor torques are NOT subtracted by this function)

	% Ix is the inertia at the center of mass of segment x
	% mx is the mass of segment x
	% cxx is the location of the CofM WRT proximal joint in the axial direction
	% cyx is the location of the CofM WRT proximal joint in the perpendicular
	% direction (Right-handed coordinate system)
	I1 = inertia.L1_I;
	I2 = inertia.L2_I;
	I3 = inertia.L3_I;
	I4 = inertia.L4_I;
	M1 = inertia.L1_M;
	M2 = inertia.L2_M;
	M3 = inertia.L3_M;
	M4 = inertia.L4_M;
	cx1 = inertia.L1_C_AXIAL;
	cx2 = inertia.L2_C_AXIAL;
	cx3 = inertia.L3_C_AXIAL;
	cx4 = inertia.L4_C_AXIAL;
	cy1 = inertia.L1_C_ANTERIOR;
	cy2 = inertia.L2_C_ANTERIOR;
	cy3 = inertia.L3_C_ANTERIOR;
	cy4 = inertia.L4_C_ANTERIOR;
	% convert to right-handed coordinate system
	if strmatch('LEFT', side, 'exact')
		cy1 = -cy1;
		cy2 = -cy2;
		cy3 = -cy3;
		cy4 = -cy4;
	end

	Im1 = data_in.([side '_KINARM']).MOTOR1_I;			%inertia of motor 1, after gear ratio, kg-m^2
	Im2 = data_in.([side '_KINARM']).MOTOR2_I;			%inertia of motor 2, after gear ratio, kg-m^2
	version = data_in.([side '_KINARM']).VERSION;
	if strncmp('KINARM_EP', version, 9)
		L1 = data_in.([side '_KINARM']).L1_L;
		delta = 0;											%angle between segments 2 and 5
	else
		% assume that it is an Exoskeleton robot.
		L1 = data_in.CALIBRATION.([side '_L1']);
		delta = data_in.([side '_KINARM']).L2_L5_ANGLE;		%angle between segments 2 and 5
	end
	L3 = data_in.([side '_KINARM']).L3_L;				%crank length (m)
	% convert to global coordinate system
	if strmatch('LEFT', side, 'exact')
		delta = - delta;
	end

	%calculate KINARM inertias relative to proximal joint
	I1_prox = I1 + M1*(cx1^2 + cy1^2);
	I2_prox = I2 + M2*(cx2^2 + cy2^2);
	I3_prox = I3 + M3*(cx3^2 + cy3^2);
	I4_prox = I4 + M4*(cx4^2 + cy4^2);

	L1Ang = data_in.([side2 '_L1Ang']);
	L2Ang = data_in.([side2 '_L2Ang']);
	L1Vel = data_in.([side2 '_L1Vel']);
	L2Vel = data_in.([side2 '_L2Vel']);
	L1Acc = data_in.([side2 '_L1Acc']);
	L2Acc = data_in.([side2 '_L2Acc']);


	theta2_1 = L2Ang - L1Ang;
	theta5_1 = theta2_1 - delta;
	sin21 = sin(theta2_1);
	cos21 = cos(theta2_1);
	sin51 = sin(theta5_1);
	cos51 = cos(theta5_1);

	A = I1_prox + I4_prox + Im1 + M2*L1^2;
	B = M2*L1*(cx2*cos21 - cy2*sin21) + M4*L3*(cx4*cos51 + cy4*sin51);
	C = M2*L1*(cx2*sin21 + cy2*cos21) + M4*L3*(cx4*sin51 - cy4*cos51);
	D = I2_prox + I3_prox + Im2 + M4*L3^2;

	% 	M = [A B; B D];	%inertial matrix
	% 	CC = [0 -C; C 0];	%coriolis and centripetal forces
	T1 = A * L1Acc + B .* L2Acc - C .* L2Vel.^2;
	T2 = B .* L1Acc + D * L2Acc + C .* L1Vel.^2;

end

