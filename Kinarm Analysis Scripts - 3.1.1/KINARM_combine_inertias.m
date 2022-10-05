function inertia_out = KINARM_combine_inertias(inertia1, inertia2)
%KINARM_COMBINE_INERTIAS Combine inertial properties from two structures.
% 	INERTIA_OUT = KINARM_COMBINE_INERTIAS(INERTIA1, INERTIA2) will combine
% 	the inertial properties of two data structures: INERTIA1 and INERTIA2.
% 	The output INERTIA_OUT will have all of the fields of INERTIA1, plus any
% 	Lx_y fields of INERTIA2, where x is 1-4 and y is I, M, C_AXIAL or
% 	C_ANTERIOR.
% 
% 	INERTIA1 and INERTIA2 should each have at least one of the following sub-fields: 
% 	L1_M, L2_M, L3_M, L4_M
% 	L1_I, L2_I, L3_I, L4_I
% 	L1_C_AXIAL, L2_C_AXIAL, L3_C_AXIAL, L4_C_AXIAL
% 	L1_C_ANTERIOR, L2_C_ANTERIOR, L3_C_ANTERIOR, L4_C_ANTERIOR
% 
% 	In each case, Ln (n = 1-4) refers to the nth link or segment of the
% 	KINARM robot (see Dexterity User Manual for more details).  
% 
% 	I is the inertia (kg-m^2) at the center of mass
% 	m is the mass (kg)
% 	C_AXIAL is the location of the center of mass from the proximal joint
% 	along the major axis if that link/segment
% 	C_ANTERIOR is the location of the center of mass, perpendicular to the
% 	main axis in the anterior direction (when KINARM robot is in the
% 	'anatomical position).  (see Dexterity User Manual for more details) 
% 
% 	For examples of using KINARM_COMBINE_INERTIAS, see the code for
% 	KINARM_ADD_TROUGH_INERTIA.

%   Copyright 2009-2021 BKIN Technologies Ltd

inertia_out = inertia1;

%make the list of field names that need to be present
names1 = fieldnames(inertia1);
names2 = fieldnames(inertia2);

for ii = 1:4
	required_fields = {['L' num2str(ii) '_M'], ['L' num2str(ii) '_I'], ['L' num2str(ii) '_C_AXIAL'], ['L' num2str(ii) '_C_ANTERIOR']};
	% test to see if any subfields for Li_ exist in either inertia1 or
	% inertia2.  If they do, then first check for existence of all required
	% Li_ subfields for both inertia1 and inertia2 and then combine them
	if ~isempty(strmatch(['L' num2str(ii) '_'], [names1; names2]));
		% Search through all the required subfields for Li_.  If they do
		% not exist then create them (setting them equal to 0).  If they do
		% exist, check that they are not empty, and if they are, then set
		% them equal to 0.  It is necessary for all fields to exist and be
		% non-empty before combining them.
		for jj = 1:length(required_fields);
			if isempty(strmatch(required_fields{jj}, names1, 'exact')) || isempty(inertia1.(required_fields{jj}))
				inertia1.(required_fields{jj}) = 0;
			end
			if isempty(strmatch(required_fields{jj}, names2, 'exact')) || isempty(inertia2.(required_fields{jj}))
				inertia2.(required_fields{jj}) = 0;
			end
		end
		
		%combine the inertia, masses and CofM for the ith segment
		C1_ANTERIOR = inertia1.(['L' num2str(ii) '_C_ANTERIOR']);
		C2_ANTERIOR = inertia2.(['L' num2str(ii) '_C_ANTERIOR']);
		C1_AXIAL	= inertia1.(['L' num2str(ii) '_C_AXIAL']);
		C2_AXIAL	= inertia2.(['L' num2str(ii) '_C_AXIAL']);
		I1			= inertia1.(['L' num2str(ii) '_I']);
		I2			= inertia2.(['L' num2str(ii) '_I']);
		M1			= inertia1.(['L' num2str(ii) '_M']);
		M2			= inertia2.(['L' num2str(ii) '_M']);
		
		M = M1 + M2;
		%the mass must be a non-zero to calculate the center of mass
		if M ~= 0		
			C_AXIAL = (C1_AXIAL * M1 + C2_AXIAL * M2) / M;
			C_ANTERIOR = (C1_ANTERIOR * M1 + C2_ANTERIOR * M2) / M;
		else
			C_AXIAL = 0;
			C_ANTERIOR = 0;
		end
		I = I1 + I2 + M1 * ((C_AXIAL - C1_AXIAL)^2 + (C_ANTERIOR - C1_ANTERIOR)^2) + M2 * ((C_AXIAL - C2_AXIAL)^2 + (C_ANTERIOR - C2_ANTERIOR)^2);

		inertia_out.(['L' num2str(ii) '_C_ANTERIOR']) = C_ANTERIOR;
		inertia_out.(['L' num2str(ii) '_C_AXIAL']) = C_AXIAL;
		inertia_out.(['L' num2str(ii) '_I']) = I;
		inertia_out.(['L' num2str(ii) '_M']) = M;

	end
end
