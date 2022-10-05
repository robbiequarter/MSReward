function DB = KINARM_create_trough_database(trough_set)

%KINARM_CREATE_TROUGH_DATABASE Create database of arm trough inertial
%parameters.
% 	DB = KINARM_CREATE_TROUGH_DATABASE(TROUGH_SET) creates a database
% 	of trough inertial properties for various KINARM Labs.  The DB
% 	output is used by the KINARM_ADD_TROUGH_INERTIA function to add arm
% 	trough inertia to the KINARM inertial properties, which can then be used
% 	by the KINARM_ADD_TORQUES function.  To create or modify a custom arm
% 	trough database, this function can be edited.
% 
%	See trough_set 'demo' in the code for an example of creating a custom
%	arm trough database set

%   Copyright 2009-2021 BKIN Technologies Ltd

if strcmp(trough_set, 'demo')
	% This is a sample database to show how a database would be created if
	% a handle was used instead of arm troughs.  
	%
	% When creating a custom database, measurements or estimates of all
	% parameters are required.  It is important to note that the dominant
	% factors for total inertia at the shoulder and elbow joints are the
	% mass (.M) and the location of the center-of-mass (CofM).  Measuring
	% .M and .C_AXIAL manually is not difficult and care should be taken to
	% measure these reasonably accurately.  In contrast, measuring the
	% inertia (.I) can be quite difficult.  But because the inertia at the
	% center of mass (I) is often <20% of the total inertia seen at the
	% joint, crude approximations for inertia at the CofM can be made (e.g.
	% using an object's mass and assuming that it is a sphere where I =
	% 2/5*m*r^2). 
	%
	% The equation for total inertia seen at the joint is:
	% Itotal = I + m * r^2, where:
	% I - inertia at CofM
	% m - mass of object
	% r = sqrt(cx^2 + cy^2), where:
	% cx = dot(db.type.location_est, [L1 1 L2 1]) + db.type.size.C_AXIAL
	% cy = db.type.size.C_ANTERIOR
	%

	% 	DB has fields for each trough_type e.g. 'HANDLE').  Each
	% 	type of trough contains sub-fields for each possible size of that
	% 	trough (e.g. 'STANDARD', 'HEAVY').  
	% 	In addition, the trough_type field must have the following fields:
	% 		.segment, which indicates which link the trough is attached to
	%			(valid options are: 'L1', 'L2', 'L3' or 'L4') 
	% 		.location_est, which indicates how to estimate the relative location of
	% 			the trough's marking line WRT the proximal joint (this measure
	% 			is relative to the clamping screw).  This estimate is only used
	% 			by KINARM_add_trough_inertia if specified to.  This is a 1x4
	% 			vector such that if: 
	%				x = DB.HANDLE.location_est, then
	%				marking_location = dot(x, [L1 1 L2 1]);  

	%	Each possible size of trough type contains sub-fields defining that 
	% 	size of that trough, including: 
	% 
	% 	.M: mass of the trough (kg)
	% 	.I: inertia of the trough at the center-of-mass (CofM) (kg-m^2)
	% 	.C_AXIAL: location (m) of the CofM along the segment axis. For the
	% 	    KINARM Classic, this measure is relative to the trough's marking
	% 	    line (i.e. small vertical line on base of trough). For the 2016
	%       Re-design, this measure is relative to the clamping screw.
	% 	.C_ANTERIOR: location (m) of the CofM perpendicular to the segment axis,
	% 	    in the 	anterior direction when in the anatomical position (i.e.
	% 	    C_ANTERIOR is the same value for left and right-handed troughs due
	% 	    to symmetry). 
	% 	.body_index_min: the minimum ratio of mass/height (kg/m) for which this
	% 		trough is typically used.  This ratio is used when guessing trough
	% 		ID (i.e. for situations in which the trough ID was not recorded).
	% 
	DB.HANDLE.segment = 'L2';
	DB.HANDLE.location_est = [0 0 1.1 0.10];
	DB.HANDLE.STANDARD = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.070, 'I', 0.0010, 'M', 0.50, 'body_index_min', 0);
	DB.HANDLE.HEAVY    = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.070, 'I', 0.0020, 'M', 10.00, 'body_index_min', 50);

elseif strcmp(trough_set, 'human_2005')
	% This database is for the SLS arm troughs, with combined
	% hand and forearm troughs, first available in fall 2005
	% for the Human KINARM Exoskeleton Classic.
	DB.UPPERARM.segment = 'L1';
	DB.UPPERARM.location_est = [0.5 0 0 0];
	DB.UPPERARM.SML = struct('C_ANTERIOR', 0, 'C_AXIAL', 0, 'I', 0.0003, 'M', 0.26, 'body_index_min', 30);
	DB.UPPERARM.MED = struct('C_ANTERIOR', 0, 'C_AXIAL', 0, 'I', 0.0003, 'M', 0.27, 'body_index_min', 40);
	DB.UPPERARM.LRG = struct('C_ANTERIOR', 0, 'C_AXIAL', 0, 'I', 0.0004, 'M', 0.28, 'body_index_min', 50);

	DB.FOREARM.segment = 'L2';
	DB.FOREARM.location_est = [0 0 0.6 -0.1];
	DB.FOREARM.SML = struct('C_ANTERIOR', 0, 'C_AXIAL', 0.062, 'I', 0.0033, 'M', 0.34, 'body_index_min', 30);
	DB.FOREARM.MED = struct('C_ANTERIOR', 0, 'C_AXIAL', 0.062, 'I', 0.0036, 'M', 0.36, 'body_index_min', 40);
	DB.FOREARM.LRG = struct('C_ANTERIOR', 0, 'C_AXIAL', 0.062, 'I', 0.0039, 'M', 0.39, 'body_index_min', 50);

elseif strcmp(trough_set, 'human_2008')
	% This database is for the urethane cast arm troughs, with separate
	% hand and forearm troughs, first available in fall 2008
	% for the Human KINARM Exoskeleton Classic.
	DB.UPPERARM.segment = 'L1';
	DB.UPPERARM.location_est = [0.5 0 0 0];
	DB.UPPERARM.XX_SML = struct('C_ANTERIOR', -0.002, 'C_AXIAL', -0.015, 'I', 0.0003, 'M', 0.28, 'body_index_min', 20);
	DB.UPPERARM.X_SML  = struct('C_ANTERIOR', -0.002, 'C_AXIAL', -0.015, 'I', 0.0003, 'M', 0.28, 'body_index_min', 25);
	DB.UPPERARM.SML    = struct('C_ANTERIOR', -0.002, 'C_AXIAL',  0.009, 'I', 0.0004, 'M', 0.31, 'body_index_min', 30);
	DB.UPPERARM.MED    = struct('C_ANTERIOR', -0.002, 'C_AXIAL',  0.009, 'I', 0.0005, 'M', 0.32, 'body_index_min', 40);
	DB.UPPERARM.LRG    = struct('C_ANTERIOR', -0.002, 'C_AXIAL',  0.008, 'I', 0.0006, 'M', 0.33, 'body_index_min', 50);

	DB.FOREARM.segment = 'L2';
	DB.FOREARM.location_est = [0 0 0.3 0];
	DB.FOREARM.XX_SML = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.000, 'I', 0.0001, 'M', 0.15, 'body_index_min', 20);
	DB.FOREARM.X_SML  = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.000, 'I', 0.0002, 'M', 0.18, 'body_index_min', 25);
	DB.FOREARM.SML    = struct('C_ANTERIOR', -0.001, 'C_AXIAL', 0.001, 'I', 0.0003, 'M', 0.17, 'body_index_min', 30);
	DB.FOREARM.MED    = struct('C_ANTERIOR', -0.003, 'C_AXIAL', 0.002, 'I', 0.0003, 'M', 0.18, 'body_index_min', 40);
	DB.FOREARM.LRG    = struct('C_ANTERIOR', -0.003, 'C_AXIAL', 0.002, 'I', 0.0004, 'M', 0.20, 'body_index_min', 50);

	DB.HAND.segment = 'L2';
	DB.HAND.location_est = [0 0 1.1 -0.20];
	DB.HAND.XX_SML  = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.027, 'I', 0.0004, 'M', 0.18, 'body_index_min', 20);
	DB.HAND.X_SML   = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.037, 'I', 0.0006, 'M', 0.20, 'body_index_min', 25);
	DB.HAND.SML     = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.072, 'I', 0.0012, 'M', 0.20, 'body_index_min', 30);
	DB.HAND.LRG     = struct('C_ANTERIOR',  0.000, 'C_AXIAL', 0.074, 'I', 0.0014, 'M', 0.22, 'body_index_min', 50);

elseif strcmp(trough_set, 'human_2016')
	% This database is for the arm troughs, with separate hand and forearm
	% troughs, first available for the re-designed Human KINARM Exoskeleton that
	% was released in fall 2016.
	DB.UPPERARM.segment = 'L1';
	DB.UPPERARM.location_est = [0.5 0.027 0 0];
	DB.UPPERARM.XX_SML = struct('C_ANTERIOR', 0.000, 'C_AXIAL', -0.027, 'I', 0.0006, 'M', 0.39, 'body_index_min', 20);
	DB.UPPERARM.X_SML  = struct('C_ANTERIOR', 0.000, 'C_AXIAL', -0.026, 'I', 0.0006, 'M', 0.35, 'body_index_min', 25);
	DB.UPPERARM.SML    = struct('C_ANTERIOR', 0.000, 'C_AXIAL', -0.025, 'I', 0.0006, 'M', 0.35, 'body_index_min', 30);
	DB.UPPERARM.MED    = struct('C_ANTERIOR', 0.000, 'C_AXIAL', -0.025, 'I', 0.0007, 'M', 0.36, 'body_index_min', 40);
	DB.UPPERARM.LRG    = struct('C_ANTERIOR', 0.000, 'C_AXIAL', -0.025, 'I', 0.0008, 'M', 0.36, 'body_index_min', 50);

	DB.FOREARM.segment = 'L2';
	DB.FOREARM.location_est = [0 0 0.3 0];
	DB.FOREARM.XX_SML = struct('C_ANTERIOR', 0.004, 'C_AXIAL', -0.002, 'I', 0.0001, 'M', 0.14, 'body_index_min', 20);
	DB.FOREARM.X_SML  = struct('C_ANTERIOR', 0.002, 'C_AXIAL',  0.000, 'I', 0.0002, 'M', 0.16, 'body_index_min', 25);
	DB.FOREARM.SML    = struct('C_ANTERIOR', 0.002, 'C_AXIAL',  0.002, 'I', 0.0003, 'M', 0.18, 'body_index_min', 30);
	DB.FOREARM.MED    = struct('C_ANTERIOR', 0.002, 'C_AXIAL',  0.002, 'I', 0.0004, 'M', 0.20, 'body_index_min', 40);
	DB.FOREARM.LRG    = struct('C_ANTERIOR', 0.000, 'C_AXIAL',  0.002, 'I', 0.0004, 'M', 0.21, 'body_index_min', 50);

	DB.HAND.segment = 'L2';
	DB.HAND.location_est = [0 0 1.1 -0.20];
	DB.HAND.X_SML   = struct('C_ANTERIOR',  0.005, 'C_AXIAL', 0.034, 'I', 0.0005, 'M', 0.18, 'body_index_min', 20);
	DB.HAND.MED     = struct('C_ANTERIOR',  0.004, 'C_AXIAL', 0.063, 'I', 0.0015, 'M', 0.25, 'body_index_min', 30);
	
else
	error('---> valid trough set not chosen for KINARM_create_trough_database');
	DB.EMPTY.EMPTY = 'empty';
end
