% Below is the basic process required to use the scripts to calculate 
% intra-muscular torques. For detailed information on each step, look at
% the help for the individual function calls.
data = zip_load('visually guided reaching 8 target.zip');

data = KINARM_add_subject_inertia(data);
data = KINARM_add_friction(data);

% use troughs for a Kinarm Exoskeleton (not the Kinarm Classic).
trough_db = KINARM_create_trough_database('human_2016');

% Estimate all of the inertias based on the subject size and trough size.
data = KINARM_add_trough_inertia(data,'trough_db', trough_db, 'trough_size', 'estimate', 'trough_location', 'estimate');

% filter AFTER adding friction, but BEFORE calculating the inverse dynamics 
data = filter_double_pass(data, 'standard', 'fc', 10);

data = KINARM_add_torques(data);