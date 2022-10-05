function dataOut = add_video_latency(dataIn, displayLatency, numBufferedFrames)
%ADD_VIDEO_LATENCY Add min/max limits to video latency
%	DATA_OUT = ADD_VIDEO_LATENCY(DATA_IN, DISPLAY_LATENCY,
%	NUM_BUFFERED_FRAMES) adds minimum and maximum times to the VIDEO_LATENCY
%	field of the structure DATA_IN.  These minimum and maximum times represent
%	the limits of when the upper-left pixel in the video frame was shown to the
%	subject. These times are based on:  
%
%		(1) the send and acknowledge times for the video frames (i.e. between
%		the Robot Computer and the Dexterit-E Computer), which provide absolute
%		time constraints on the command sequence. These calculations also
%		include some 'intelligence' which corrects acknowledgement times based
%		on the fact that the minimum time between the actual display of an image
%		is the video frame period.   
%
%		(2) buffering by the subject display, if present (most modern displays
%		buffer the entire image before displaying it, whereas CRT-like displays
%		have no buffer) 
%
%		(3) other latencies in the display (which includes the "response time"
%		reported by the display manufacturer, but can also include other delays
%		if they exist. Please see the Dexterit-E User Guide reference section
%		for more information).   
%
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = add_video_latency(data)
%     OR
%   dataNew.c3d = add_video_latency(data(ii).c3d)
%
%   Copyright 2009-2021 BKIN Technologies Ltd
%	The DISPLAY_LATENCY input is a required input (in seconds), and should be a
%	measure of all delays inherent in the display, excluding any buffering by
%	the display. The typical sources of delay are: (i) response time (e.g. 5-10
%	ms); (ii)asynchronous backlight pulse-width-modulation (0 if none, or 4 ms
%	if 120 Hz PWM); (iii) internal processing delays (0-5 ms) 
%
%	The NUM_BUFFERED_FRAMES is an optional input which indicates the number of
%	frames that the display buffers. The default value is 1 (i.e. if no argument
%	is input to the function) . For most modern displays, this value = 1. For
%	CRT or CRT-like displays, this value should be set = 0. 
%
%   Note 1: under normal conditions (i.e. in which the communication between the
%   Robot and Dexterit-E computers is not delayed via parallel or conflicting
%   process), the maximum time represents the actual time of the display (i.e.
%   typically the acknowledgement is received within ~1 ms of the Vsync pulse on
%   the video card displaying the image).  
%
%	Note 2: this function does NOT account for discrepancies in timing of visual
%	stimulus on different parts of the screen. Most modern displays behave in a
%	manner similar to CRTs: the frame is displayed sequentially, line-by-line,
%	from top to bottom, over the duration of an entire single frame. An example
%	of an exception to this behaviour are many DLP projectors which display the
%	entire frame synchronously, but which display the RGB colours sequentially.

%   Copyright 2009-2021 BKIN Technologies Ltd

if nargin==0
	error('---> No input provided ');
elseif nargin == 1 || isempty(displayLatency) || ~isnumeric(displayLatency) 
	error('---> No display_latency was specified, or was specified improperly. Must be a numeric value for display device latency (specified in seconds). ');
elseif nargin == 2
	numBufferedFrames = 1;
elseif isempty(numBufferedFrames) || ~isnumeric(numBufferedFrames) 
	error('---> num_buffered_frames was specified improperly. Must be a numeric value (specified in frames). ');
end

if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then add video latency to each exam file, one at a time.
	dataOut = dataIn;
	for jj = 1:length(dataIn)
		dataOut(jj).c3d = AddVideoLatencyToAllTrials(dataIn(jj).c3d, displayLatency, numBufferedFrames);
	end
else
	% legacy functionality, assuming that data_in = examFile(ii).c3d
	dataOut = AddVideoLatencyToAllTrials(dataIn, displayLatency, numBufferedFrames);
end	    

disp('Finished adding video latency estimates.');


function data = AddVideoLatencyToAllTrials(data, displayLatency, numBufferedFrames)

% for each trial of data in dataIn
for ii = 1:length(data)
	if ~isempty(data(ii).VIDEO_LATENCY)
		videoFramePeriod = 1/data(ii).VIDEO_SETTINGS.REFRESH_RATE;				%reported video refresh rate in sec (Dexterit-E computer clock)
		bufferDelay = numBufferedFrames * videoFramePeriod;
		
		%reported video refresh rate is rounded to the nearest ms, and differences
		%of a few percent between Dexterit-E and Robot computer is possible.
		%videoFramePeriodFloor puts minimum limit on the different that could be
		%recorded by the Robot computer.
		videoFramePeriodFloor = 0.001 * floor( 0.95* videoFramePeriod *1000);						
		ackTimeCorrected = data(ii).VIDEO_LATENCY.ACK_TIMES;

		% The following correction is based on the fact that the time between video
		% display refresh is the videoFramePeriod, which on the Robot computer
		% has a minimum expected value of videoFramePeriodFloor.  As such, the time
		% between any two adjacent video acknowledgements must be
		% >= videoFramePeriodFloor.  Any discrepancies with this fact are corrected
		% here.
		for jj = length(ackTimeCorrected):-1:2
			if (ackTimeCorrected(jj) - ackTimeCorrected(jj-1)) < videoFramePeriodFloor	%use of floor ensures that only those periods 
				ackTimeCorrected(jj - 1) = ackTimeCorrected(jj) - videoFramePeriodFloor;
			end
		end

		% The minimum and maximum video latencies are calculated from the SEND and
		% corrected acknowledgement times, plus the following:
		% (1) the addition of a refresh period (required to transmit the image)
		% (2) the display_latency

		data(ii).VIDEO_LATENCY.DISPLAY_MIN_TIMES = data(ii).VIDEO_LATENCY.SEND_TIMES + bufferDelay + displayLatency;
		data(ii).VIDEO_LATENCY.DISPLAY_MAX_TIMES = ackTimeCorrected + bufferDelay + displayLatency;
	else
		% no video latency data, so do nothing
	end
end
