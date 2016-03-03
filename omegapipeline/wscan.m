function wscan(eventTime, uniqueID, configurationFile, frameCacheFile, ...
               outputDirectory, generateReport, debugLevel)
% WSCAN Create spectrograms of significant channels for candidate events
%
% WSCAN Examines gravitational-wave channels, auxiliary interferometer channels,
% and environmental channels around times of interest.  Spectrograms are
% generated for those channels that exhibit statistically significant behavior.
%
% usage: wscan(eventTime, configurationFile, frameCacheFile, ...
%              outputDirectory, generateReport, uniqueid, debugLevel);
%
%   eventTime             GPS time of candidate event
%   configurationFile     path name of channel configuration file
%   frameCacheFile        path name of frame file cache file
%   outputDirectory       directory to write results
%   generateReport        generate plots and html report
%   uniqueID		  glitch ID tag
%   debugLevel            verboseness of debug level output
%
% To allow use as a stand-alone executable, the specified eventTime should be
% a string, not a number.
%
% The configuration file is an ASCII text file describing the parameters for
% each channel to be analyzed.  The entries should have the following syntax.
%
% {
%   channelName:                 'H1:LSC-AS_Q'
%   frameType:                   'RDS_R_L1'
%   sampleFrequency:             4096
%   searchTimeRange:             16
%   searchFrequencyRange:        [64 1024]
%   searchQRange:                [4 64]
%   searchMaximumEnergyLoss:     0.2
%   whiteNoiseFalseRate:         1e-2
%   searchWindowDuration:        0.1
%   plotTimeRanges:              [0.1 1.0 10.0]
%   plotFrequencyRange:          [64 1024]
%   plotNormalizedEnergyRange:   [0 25.5]
%   alwaysPlotFlag:              0
% }
%
% Groups of related channels may optionally be separated by a section title
% specifier with the following form.
%
% [index_entry:section_title]
%
% This will be used to generate a index entry and section title in the resulting
% web page.
%
% The WCONFIGURE.SH script can be used to automatically generate a reasonable
% configuration file for sample frame files.
%
% If no configuration file is specified, WSCAN looks for the file
% configuration.txt in the current working directory.  Similarly, if not frame
% cache file is specified, WSCAN looks for the file framecache.txt in the
% current working directory.
%
% For information on the frameCacheFile, see READFRAMEDATA.
%
% The resulting spectrograms for statistically significant channels are written
% to an event sub-directory that is created in the specified output directory
% and is labelled by the GPS time of the event under analysis.  If no output
% directory is specified, a default output directory called wscans is created in
% the current working directory.  A web page named index.html is also created in
% each event sub-directory and presents the resulting spectrograms in table
% form.
%
% The specified debugLevel controls the amount of detail in the output log.
% A debugLevel of unity is assumed by default.
%
% See also WREADDATA, WRESAMPLE, WTILE, WCONDITION, WTRANSFORM, WTHRESHOLD,
% WSELECT, WSPECTROGRAM, WEVENTGRAM, WTIMESERIES, WCONFIGURE.SH and WSCAN.SH.

% Shourov K. Chatterji <shourov@ligo.mit.edu>

% $Id: wscan.m 3431 2013-11-02 22:35:51Z brennan.hughey@LIGO.ORG $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                   defaults                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% default configuration parameters
defaultSampleFrequency = 4096;
defaultSearchTimeRange = 64;
defaultSearchFrequencyRange = [0 Inf];
defaultSearchQRange = [4 64];
defaultSearchMaximumEnergyLoss = 0.2;
defaultWhiteNoiseFalseRate = 1e-2;
defaultSearchWindowDuration = 0.1;
defaultPlotTimeRanges = [1 4 16];
defaultPlotFrequencyRange = [];
defaultPlotNormalizedEnergyRange = [0 25.5];
defaultAlwaysPlotFlag = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            hard coded parameters                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% search parameters
transientFactor = 2;
outlierFactor = 2.0;

% display parameters
plotHorizontalResolution = 512;
plotDurationInflation = 0.5;
plotBandwidthInflation = 0.5;

% limits on number of significant tiles
maximumSignificants = 1e4;
maximumMosaics = 1e4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        process command line arguments                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% verify correct number of input arguments
narginchk(1, 7);

% apply default arguments
if (nargin < 3) || isempty(configurationFile),
  configurationFile = 'configuration.txt';
end
if (nargin < 4) || isempty(frameCacheFile),
  frameCacheFile = 'framecache.txt';
end
if (nargin < 5)
  outputDirectory = [];
end
if (nargin < 6) || isempty(generateReport),
  generateReport = false;
end
if (nargin < 7) || isempty(debugLevel),
  debugLevel = 1;
end

% convert string event time and debug level to numbers
if ischar(eventTime),
  %eventTimeString = eventTime;
  eventTime = str2num(eventTime);
else
  %eventTimeString = sprintf('%#020.9f', eventTime);
end
if ischar(generateReport),
  generateReport = logical(str2num(generateReport));
end
if ischar(debugLevel),
  debugLevel = str2num(debugLevel);
end

% name of html file
htmlFile = 'index.html';

% name of text summary file
textSummaryFile = 'summary.txt';

% name of xml summary file
xmlSummaryFile = 'summary.xml';

% name of context file
contextFile = 'context.html';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 write header                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'OmegaScan of event at %s\n', uniqueID);
wlog(debugLevel, 1, 'created by %s on %s at %s\n', ...
     getenv('USER'), datestr(clock, 29), datestr(clock, 13));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       initialize random number generators                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set random number generator seeds based on event time
rand('state', eventTime);
randn('state', eventTime);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           read configuration file                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'reading configuration file %s...\n', configurationFile);

% enable or disable warnings
if debugLevel >= 2,
  warning on WSCAN:incompleteConfiguration
else
  warning off WSCAN:incompleteConfiguration
end

% initialize configuration structure
configuration = [];
channelNumber = 0;
sectionIndex = [];
sectionName = [];
sectionNumber = 0;
sectionStart = [];

% open configuration file for reading
configurationFID = fopen(configurationFile, 'r');

% begin loop over configuration file
while ~feof(configurationFID),

  % read one line from configuration file
  configurationLine = fgetl(configurationFID);

  % remove any comments
  commentIndices = strfind(configurationLine, '#');
  if ~isempty(commentIndices),
    configurationLine = configurationLine(1 : (commentIndices(1) - 1));
  end

  % remove leading and trailing blanks
  configurationLine = strtrim(configurationLine);

  % if empty line, skip to the next line
  if isempty(configurationLine),
    continue;
  end

  % check for new section
  if configurationLine(1) == '[',

    % locate field separator
    commaIndex = strfind(configurationLine, ',');

    % if field separator not located, report syntax error
    if isempty(commaIndex),
      error('syntax error processing configuration file "%s":\n%s\n', ...
            configurationFile, configurationLine);
    end

    % select first field separator
    commaIndex = commaIndex(1);

    % increment section number
    sectionNumber = sectionNumber + 1;

    % exract section index
    sectionIndex{sectionNumber} = configurationLine(2 : commaIndex - 1);

    % extract section name
    sectionName{sectionNumber} = configurationLine((commaIndex + 1) : end - 1);

    % remove leading and trailing blanks
    sectionIndex{sectionNumber} = ...
        fliplr(deblank(fliplr(deblank(sectionIndex{sectionNumber}))));
    sectionName{sectionNumber} = ...
        fliplr(deblank(fliplr(deblank(sectionName{sectionNumber}))));

    % standardize section names
    sectionIndex{sectionNumber} = strrep(sectionIndex{sectionNumber}, ';', ':');
    sectionName{sectionNumber} = strrep(sectionName{sectionNumber}, ';', ':');

    % determine initial visibility
    switch sectionIndex{sectionNumber},
      case 'Context',
        sectionChecked{sectionNumber} = 'checked';
        sectionDisplay{sectionNumber} = 'block';
      case 'Gravitational',
        sectionChecked{sectionNumber} = 'checked';
        sectionDisplay{sectionNumber} = 'block';
      otherwise
        sectionChecked{sectionNumber} = 'checked';
        sectionDisplay{sectionNumber} = 'block';
        % sectionChecked{sectionNumber} = '';
        % sectionDisplay{sectionNumber} = 'none';
    end

    % record first channel in section
    sectionStart(sectionNumber) = channelNumber + 1;

    % continue to next line
    continue;

  end

  % check for beginning of new channel configuration
  if configurationLine == '{',

    % increment channel number
    channelNumber = channelNumber + 1;

    % initialize configuration parameters
    configuration{channelNumber}.channelName = [];
    configuration{channelNumber}.frameType = [];
    configuration{channelNumber}.sampleFrequency = [];
    configuration{channelNumber}.searchTimeRange = [];
    configuration{channelNumber}.searchFrequencyRange = [];
    configuration{channelNumber}.searchQRange = [];
    configuration{channelNumber}.searchMaximumEnergyLoss = [];
    configuration{channelNumber}.searchWindowDuration = [];
    configuration{channelNumber}.plotTimeRanges = [];
    configuration{channelNumber}.plotFrequencyRange = [];
    configuration{channelNumber}.plotNormalizedEnergyRange = [];
    configuration{channelNumber}.alwaysPlotFlag = [];

    % continue to next line
    continue;

  end

  % check for end of existing channel configuration
  if configurationLine == '}',

    % validate channel configuration
    if isempty(configuration{channelNumber}.channelName),
      error(['channel name not specified for ' ...
             'channel number %d'], channelNumber);
    end
    if isempty(configuration{channelNumber}.frameType),
      error(['frame type not specified for ' ...
             'channel number %d'], channelNumber);
    end
    if isempty(configuration{channelNumber}.sampleFrequency),
      warning('WSCAN:incompleteConfiguration', ...
              ['sample frequency not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.sampleFrequency = ...
          defaultSampleFrequency;
    end
    if isempty(configuration{channelNumber}.searchTimeRange),
      warning('WSCAN:incompleteConfiguration', ...
              ['search time range not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchTimeRange = ...
          defaultSearchTimeRange;
    end
    if any(rem(configuration{channelNumber}.searchTimeRange,1) ~= 0),
      error('WSCAN:invalidConfiguration', ...
            ['search time range not specified as integers for ' ...
             'channel number %d'], channelNumber);
    end
    if isempty(configuration{channelNumber}.searchFrequencyRange),
      warning('WSCAN:incompleteConfiguration', ...
              ['search frequency range not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchFrequencyRange = ...
          defaultSearchFrequencyRange;
    end
    if isempty(configuration{channelNumber}.searchQRange),
      warning('WSCAN:incompleteConfiguration', ...
              ['search Q range not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchQRange = ...
          defaultSearchQRange;
    end
    if isempty(configuration{channelNumber}.searchMaximumEnergyLoss),
      warning('WSCAN:incompleteConfiguration', ...
              ['search maximum energy loss not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchMaximumEnergyLoss = ...
          defaultSearchMaximumEnergyLoss;
    end
    if isempty(configuration{channelNumber}.whiteNoiseFalseRate),
      warning('WSCAN:incompleteConfiguration', ...
              ['white noise false rate not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchMaximumEnergyLoss = ...
          defaultWhiteNoiseFalseRate;
    end
    if isempty(configuration{channelNumber}.searchWindowDuration),
      warning('WSCAN:incompleteConfiguration', ...
              ['search window duration not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.searchWindowDuration = ...
          defaultSearchWindowDuration;
    end
    if isempty(configuration{channelNumber}.plotTimeRanges),
      warning('WSCAN:incompleteConfiguration', ...
              ['plot time ranges not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.plotTimeRanges = ...
          defaultPlotTimeRanges;
    end
    if isempty(configuration{channelNumber}.plotFrequencyRange),
      warning('WSCAN:incompleteConfiguration', ...
              ['plot frequency range not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.plotFrequencyRange = ...
          defaultPlotFrequencyRange;
    end
    if isempty(configuration{channelNumber}.plotNormalizedEnergyRange),
      warning('WSCAN:incompleteConfiguration', ...
              ['plot normalized energy range not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.plotNormalizedEnergyRange = ...
          defaultPlotNormalizedEnergyRange;
    end
    if isempty(configuration{channelNumber}.alwaysPlotFlag),
      warning('WSCAN:incompleteConfiguration', ...
              ['always plot flag not specified for ' ...
               'channel number %d'], channelNumber);
      configuration{channelNumber}.alwaysPlotFlag = ...
          defaultAlwaysPlotFlag;
    end

    % continue to next line
    continue;

  end

  % locate field separator
  colonIndex = strfind(configurationLine, ':');

  % if field separator not located, report syntax error
  if isempty(colonIndex),
    error('syntax error processing configuration file "%s":\n%s\n', ...
          configurationFile, configurationLine);
  end

  % parse configuration line
  colonIndex = colonIndex(1);
  parameterName = configurationLine(1 : colonIndex);
  parameterValue = configurationLine((colonIndex + 1) : end);
  parameterName = fliplr(deblank(fliplr(deblank(parameterName))));
  parameterValue = fliplr(deblank(fliplr(deblank(parameterValue))));

  % assign parameters based on name
  switch parameterName
    case 'channelName:'
      configuration{channelNumber}.channelName = ...
          eval(parameterValue);
    case 'frameType:'
      configuration{channelNumber}.frameType = ...
          eval(parameterValue);
    case 'sampleFrequency:'
      configuration{channelNumber}.sampleFrequency = ...
          eval(parameterValue);
    case 'searchTimeRange:'
      configuration{channelNumber}.searchTimeRange = ...
          eval(parameterValue);
    case 'searchFrequencyRange:'
      configuration{channelNumber}.searchFrequencyRange = ...
          eval(parameterValue);
    case 'searchQRange:'
      configuration{channelNumber}.searchQRange = ...
          eval(parameterValue);
    case 'searchMaximumEnergyLoss:'
      configuration{channelNumber}.searchMaximumEnergyLoss = ...
          eval(parameterValue);
    case 'whiteNoiseFalseRate:'
      configuration{channelNumber}.whiteNoiseFalseRate = ...
          eval(parameterValue);
    case 'searchWindowDuration:'
      configuration{channelNumber}.searchWindowDuration = ...
          eval(parameterValue);
    case 'plotTimeRanges:'
      configuration{channelNumber}.plotTimeRanges = ...
          eval(parameterValue);
    case 'plotFrequencyRange:'
      configuration{channelNumber}.plotFrequencyRange = ...
          eval(parameterValue);
    case 'plotNormalizedEnergyRange:'
      configuration{channelNumber}.plotNormalizedEnergyRange = ...
          eval(parameterValue);
    case 'alwaysPlotFlag:'
      configuration{channelNumber}.alwaysPlotFlag = ...
          eval(parameterValue);
    otherwise
      error(['unknown configuration parameter ' parameterName]);
  end

% end loop over channel configuration file
end

% close configuration file
fclose(configurationFID);
toc;
% number of configured channels
numberOfChannels = length(configuration);

% number of sections
numberOfSections = length(sectionName);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            load frame cache file                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'reading framecache file %s...\n', frameCacheFile);

% load frame file cache
frameCache = loadframecache(frameCacheFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           create output directory                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if outputDirectory not specified, make one based on center time
if isempty(outputDirectory),
  outputDirectory = sprintf('scans/%s', uniqueID);
end

% report status
wlog(debugLevel, 1, 'creating event directory\n');
wlog(debugLevel, 1, '  outputDirectory:         %s\n', outputDirectory);

% create spectrogram directory
unix(['mkdir -p ' outputDirectory]);

% copy configuration file
unix(['cp ' configurationFile ' ' outputDirectory '/configuration.txt']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        initialize text summary report                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'opening text summary...\n');

% open text summary file
textSummaryFID = fopen([outputDirectory '/' textSummaryFile], 'w');

% write column definitions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            initialize html report                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if generateReport,

% report status
wlog(debugLevel, 1, 'opening html report...\n');

% open web page
htmlFID = fopen([outputDirectory '/' htmlFile], 'w');

% begin html
fprintf(htmlFID, '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n');
fprintf(htmlFID, '<html>\n');
fprintf(htmlFID, '<head>\n');
fprintf(htmlFID, '<title>Scan %s</title>\n', uniqueID);
fprintf(htmlFID, '<link rel="icon" type="image/x-icon" href="omega.logo.icon.png" />\n');
fprintf(htmlFID, '<link rel="stylesheet" type="text/css" href="wstyle.css" />\n');
fprintf(htmlFID, '<script type="text/javascript">\n');
fprintf(htmlFID, 'function showImage(gpsTime, channelName, timeRanges, imageType) {\n');
fprintf(htmlFID, '  for (var timeRangeIndex in timeRanges) {\n');
fprintf(htmlFID, '    var imageBaseName =\n');
fprintf(htmlFID, '      gpsTime + "_" + channelName + "_" + timeRanges[timeRangeIndex];\n');
fprintf(htmlFID, '    document.getElementById("a_" + imageBaseName).href =\n');
fprintf(htmlFID, '      imageBaseName + "_" + imageType + ".png";\n');
fprintf(htmlFID, '    document.getElementById("img_" + imageBaseName).src =\n');
fprintf(htmlFID, '      imageBaseName + "_" + imageType + ".thumb.png";\n');
fprintf(htmlFID, '  }\n');
fprintf(htmlFID, '}\n');
fprintf(htmlFID, 'function toggleVisible(division) {\n');
fprintf(htmlFID, '  if (document.getElementById("div_" + division).style.display == "none") {\n');
fprintf(htmlFID, '    document.getElementById("div_" + division).style.display = "block";\n');
fprintf(htmlFID, '    document.getElementById("input_" + division).checked = true;\n');
fprintf(htmlFID, '  } else {\n');
fprintf(htmlFID, '    document.getElementById("div_" + division).style.display = "none";\n');
fprintf(htmlFID, '    document.getElementById("input_" + division).checked = false;\n');
fprintf(htmlFID, '  } \n');
fprintf(htmlFID, '}\n');
fprintf(htmlFID, 'function gotoSection(section) {\n');
fprintf(htmlFID, '  document.getElementById("div_" + section).style.display = "block";\n');
fprintf(htmlFID, '  document.getElementById("input_" + section).checked = true;\n');
fprintf(htmlFID, '  window.location.hash = section;\n');
fprintf(htmlFID, '}\n');
fprintf(htmlFID, '</script>\n');
fprintf(htmlFID, '</head>\n');
fprintf(htmlFID, '<body>\n');
fprintf(htmlFID, '\n');

fprintf(htmlFID, '<h1 class="title"><a href="https://www.zooniverse.org/projects/zooniverse/gravity-spy" class="title">Gravity Spy Omega Pipeline</a> Scan</h1>\n');
fprintf(htmlFID, '<h1>Scan %s</h1>\n', uniqueID);
fprintf(htmlFID, '\n');
fprintf(htmlFID, '<div class="main">\n');
fprintf(htmlFID, '\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           add index to html report                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(htmlFID, '<h2>Index</h2>\n');
fprintf(htmlFID, '<div class="section">\n');
fprintf(htmlFID, '<ul>\n');
for sectionNumber = 1 : numberOfSections,
  fprintf(htmlFID, ...
          '<li><a href="javascript:gotoSection(''%s'')">%s</a></li>\n', ...
          sectionIndex{sectionNumber}, sectionIndex{sectionNumber});
end
fprintf(htmlFID, '<li><a href="log.txt">Scan Log</a></li>\n');
fprintf(htmlFID, '<li><a href="%s">Text Summary</a></li>\n', textSummaryFile);
fprintf(htmlFID, '<li><a href="%s">XML Summary</a></li>\n', xmlSummaryFile);
fprintf(htmlFID, '<li><a href="configuration.txt">Configuration</a></li>\n');
fprintf(htmlFID, '</ul>\n');
fprintf(htmlFID, '</div>\n');
fprintf(htmlFID, '\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           initialize figure handle                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figureHandle = figure;

end % generateReport

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       loop over configuration channels                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize section status
inSectionFlag = 0;
toc;
% begin loop over channel configurations
for channelNumber = 1 : numberOfChannels,

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        add section to html report                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if generateReport,

  % check for start of new section
  sectionNumbers = find(sectionStart == channelNumber);

  % if start of new section
  if ~isempty(sectionNumbers),

    % begin loop over new sections
    for sectionNumber = sectionNumbers,

      % end any previously existing section
      if inSectionFlag == 1,
        fprintf(htmlFID, '</div>\n');
        fprintf(htmlFID, '\n');
      end

      % label section
      fprintf(htmlFID, '<a name="%s"></a>\n', sectionIndex{sectionNumber});
      fprintf(htmlFID, '<h2 class="%s">\n', sectionIndex{sectionNumber});

      fprintf(htmlFID, '<input id="input_%s" type="checkbox" %s\n', ...
              sectionIndex{sectionNumber}, sectionChecked{sectionNumber});
      fprintf(htmlFID, '       onclick="toggleVisible(''%s'');" />\n', ...
              sectionIndex{sectionNumber});
      fprintf(htmlFID, '%s\n', sectionName{sectionNumber});
      fprintf(htmlFID, '</h2>\n');
      fprintf(htmlFID, '\n');
      fprintf(htmlFID, '<div class="section" id="div_%s" style="display: %s;">\n', ...
              sectionIndex{sectionNumber}, sectionDisplay{sectionNumber});
      fprintf(htmlFID, '\n');

      % record start of section
      inSectionFlag = 1;

      % transcribe context information
      if any(strcmp(sectionIndex{sectionNumber}, {'Context', 'Timing'})),
        contextFile = [outputDirectory '/' contextFile];
        if exist(contextFile,'file'),
          wlog(debugLevel, 1, '  adding context file...\n');

          fprintf(htmlFID, '<!-- BEGIN CONTEXT -->\n');
          fprintf(htmlFID, '\n');
          context = dataread('file', contextFile, '%s', ...
                             'delimiter', '\n');
          for lineNumber = 1 : length(context),
            fprintf(htmlFID, '%s\n', context{lineNumber});
          end
          fprintf(htmlFID, '\n');
          fprintf(htmlFID, '<!-- END CONTEXT -->\n');
          fprintf(htmlFID, '\n');
          clear context;
        else
          wlog(debugLevel, 1, '  context file not found. skipping...\n');
        end
      elseif (strcmp(sectionIndex{sectionNumber}, 'Parameters')),
        fprintf(htmlFID, '<!-- BEGIN PARAMETERS -->\n');
        fprintf(htmlFID, '\n');
        fprintf(htmlFID, '<!-- END PARAMETERS -->\n');
        fprintf(htmlFID, '\n');
      elseif (strcmp(sectionIndex{sectionNumber}, 'Notes')),
        fprintf(htmlFID, '<!-- BEGIN NOTES -->\n');
        fprintf(htmlFID, '\n');
        fprintf(htmlFID, '<!-- END NOTES -->\n');
        fprintf(htmlFID, '\n');
      end

    % end loop over new sections
    end

  % otherwise, continue
  end

  end % generateReport
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %               identify statistically significant channels                  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % extract search configuration for channel
  channelName = configuration{channelNumber}.channelName;
  frameType = configuration{channelNumber}.frameType;
  sampleFrequency = configuration{channelNumber}.sampleFrequency;
  timeRange = configuration{channelNumber}.searchTimeRange;
  frequencyRange = configuration{channelNumber}.searchFrequencyRange;
  qRange = configuration{channelNumber}.searchQRange;
  maximumEnergyLoss = configuration{channelNumber}.searchMaximumEnergyLoss;
  searchWindowDuration = configuration{channelNumber}.searchWindowDuration;
  whiteNoiseFalseRate = configuration{channelNumber}.whiteNoiseFalseRate;
  alwaysPlotFlag = configuration{channelNumber}.alwaysPlotFlag;

  % string name of channel
  if iscell(channelName),
    channelNameString = ...
        strrep(deblank(sprintf('%s ', channelName{:})), ' ', '-');
  else
    channelNameString = channelName;
  end

  % standardize channel names
  channelNameString = strrep(channelNameString, ';', ':');

  % display status
  wlog(debugLevel, 1, 'processing channel %s...\n', channelNameString);
  
  % find closest sample time to event time
  centerTime = floor(eventTime) + ...
               round((eventTime - floor(eventTime)) * ...
                     sampleFrequency) / sampleFrequency;

  % determine segment start and stop times
  startTime = round(centerTime - timeRange / 2);
  stopTime = startTime + timeRange;

  % generate search tiling
  wlog(debugLevel, 2, '  tiling for search...\n');
  highPassCutoff = [];
  lowPassCutoff = [];
  whiteningDuration = [];
  tiling = wtile(timeRange, qRange, frequencyRange, sampleFrequency, ...
                 maximumEnergyLoss, highPassCutoff, lowPassCutoff, ...
                 whiteningDuration, transientFactor);

  % read data from frame file
  wlog(debugLevel, 2, '  reading data...\n');
  timeShifts = [];
  [rawData, rawSampleFrequency] = ...
      wreaddata(frameCache, channelName, frameType, ...
                startTime, stopTime, timeShifts, debugLevel);

  % check for read error
  if ~all(rawSampleFrequency),
    wlog(debugLevel, 1, '  ERROR: cannot load frame data\n');
    continue
  end

  % test multiple channels for consistency
  if length(rawData) > 1,
    wlog(debugLevel, 2, '  subtracting data...\n');
    rawData{1} = rawData{1} - rawData{2};
    rawData = rawData(1);
    rawSampleFrequency = rawSampleFrequency(1);
  end

  % check for all zeros
  if (min(rawData{1}) == max(rawData{1})),
    wlog(debugLevel, 1, '  WARNING: channel contains no information\n');
    continue
  end

  % resample data
  wlog(debugLevel, 2, '  resampling data...\n');
  rawData = wresample(rawData, rawSampleFrequency, sampleFrequency);

  % high pass filter and whiten data
  wlog(debugLevel, 2, '  conditioning data...\n');
  [~, ~, whitenedData] = ...
      wcondition(rawData, tiling);

  % q transform whitened data
  wlog(debugLevel, 2, '  transforming whitened data...\n');
  whitenedTransform = ...
      wtransform(whitenedData, tiling, outlierFactor, [], channelNameString);

  % identify most significant whitened transform tile
  wlog(debugLevel, 2, '  measuring peak significance...\n');
  thresholdReferenceTime = centerTime;
  thresholdTimeRange = 0.5 * searchWindowDuration * [-1 +1];
  thresholdFrequencyRange = [];
  thresholdQRange = [];
  whitenedProperties = ...
      wmeasure(whitenedTransform, tiling, startTime, thresholdReferenceTime, ...
               thresholdTimeRange, thresholdFrequencyRange, thresholdQRange, ...
               debugLevel);
  
  % determine normalized energy threshold
  normalizedEnergyThreshold = ...
      -log(whiteNoiseFalseRate * tiling.duration / ...
           (1.5 * tiling.numberOfIndependents));

  % if channel not significant
  if whitenedProperties{1}.peakNormalizedEnergy < normalizedEnergyThreshold,

    % if always plot flag is not set
    if alwaysPlotFlag == 0,

      % report status and skip to the next channel
      wlog(debugLevel, 1, ['  channel not significant at white noise false rate ' ...
                          '%7.1e\n'], whiteNoiseFalseRate);
      fprintf(textSummaryFID, ...
              '%-30s %#020.9f %#09.3e %#09.3e %#09.3e %#09.3e\n', ...
              channelNameString, 0, 0, 0, 0, 0);
      continue

    % end test for always plot flag
    end

  % end test of channel significance
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                analyze statistically significant channels                  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % q transform raw data
  %wlog(debugLevel, 2, '  transforming raw data...\n');
  %rawTransform = ...
  %    wtransform(rawData, tiling, outlierFactor, [], channelNameString);

  if generateReport,

  % extract plot configuration for channel
  plotTimeRanges = configuration{channelNumber}.plotTimeRanges;
  plotFrequencyRange = configuration{channelNumber}.plotFrequencyRange;
  plotNormalizedEnergyRange = ...
      configuration{channelNumber}.plotNormalizedEnergyRange;

  % identify significant whitened q transform tiles
  wlog(debugLevel, 2, '  thresholding whitened transform...\n');
  thresholdReferenceTime = centerTime;
  thresholdTimeRange = 0.5 * [-1 +1] * ...
      (max(plotTimeRanges) + ...
       tiling.planes{end}.rows{1}.duration * plotDurationInflation);
  thresholdFrequencyRange = plotFrequencyRange;
  thresholdQRange = [];
  whitenedSignificants = ...
      wthreshold(whitenedTransform, tiling, startTime, whiteNoiseFalseRate, ...
                 thresholdReferenceTime, thresholdTimeRange, ...
                 thresholdFrequencyRange, thresholdQRange, ...
                 maximumSignificants, [], [], [], [], ...
                 debugLevel);
  
  % identify significant raw q transform tiles
  %wlog(debugLevel, 2, '  thresholding raw transform...\n');
  %rawSignificants = ...
  %    wthreshold(rawTransform, tiling, startTime, whiteNoiseFalseRate, ...
  %               thresholdReferenceTime, thresholdTimeRange, ...
  %               thresholdFrequencyRange, thresholdQRange, ...
  %               maximumSignificants, [], [], [], [], ...
  %               debugLevel);

  % select non-overlapping whitened significant tiles
  wlog(debugLevel, 2, '  selecting whitened events...\n');
  whitenedSignificants = wselect(whitenedSignificants, ...
                                 plotDurationInflation, ...
                                 plotBandwidthInflation, ...
                                 maximumMosaics, debugLevel);

  % select non-overlapping raw significant tiles
  %wlog(debugLevel, 2, '  selecting raw events...\n');
  %rawSignificants = wselect(rawSignificants, ...
  %                          plotDurationInflation, ...
  %                          plotBandwidthInflation, ...
  %                          maximumMosaics, debugLevel);

  % recover time series data
  %rawData = wifft(rawData);
  %highPassedData = wifft(highPassedData);
  %whitenedData = wifft(whitenedData);

  end % generateReport
  toc;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                      add channel to summary report                         %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % identify most significant raw transform tile
  %wlog(debugLevel, 2, '  measuring peak properties...\n');
  %thresholdReferenceTime = ...
  %    whitenedProperties{1}.peakTime;
  %thresholdTimeRange = ...
  %    whitenedProperties{1}.peakDuration * [-1 +1];
  %thresholdFrequencyRange = ...
  %    whitenedProperties{1}.peakFrequency + ...
  %    whitenedProperties{1}.peakBandwidth * [-1 +1];
  %thresholdQRange = [];
  %rawProperties = ...
  %    wmeasure(rawTransform, tiling, startTime, thresholdReferenceTime, ...
  %             thresholdTimeRange, thresholdFrequencyRange, thresholdQRange, ...
  %             debugLevel);
  
  % most significant tile properties
  %mostSignificantTime = ...
  %    whitenedProperties{1}.peakTime;
  mostSignificantFrequency = ...
      whitenedProperties{1}.peakFrequency;
  mostSignificantQ = ...
      whitenedProperties{1}.peakQ;
  mostSignificantNormalizedEnergy = ...
      whitenedProperties{1}.peakNormalizedEnergy;
  % mostSignificantAmplitude = ...
  %     rawProperties{1}.peakAmplitude;
  %mostSignificantAmplitude = ...
  %    rawProperties{1}.signalAmplitude;

  % write most significant tile properties to text summary file
  wlog(debugLevel, 2, '  writing text summary...\n');
  fprintf(textSummaryFID, ...
          '%-30s %#020.9f %#09.3e %#09.3e %#09.3e\n', ...
          channelNameString, uniqueID, mostSignificantFrequency, ...
          mostSignificantQ, mostSignificantNormalizedEnergy ...
          );


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        add channel to html report                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if generateReport,

  % html for most significant tile properties
  %mostSignificantTimeHtml = ...
  %    sprintf('%.3f', mostSignificantTime);
  mostSignificantFrequencyHtml = ...
      sprintf('%.1f', mostSignificantFrequency);
  mostSignificantQHtml = ...
      sprintf('%.1f', mostSignificantQ);
  mostSignificantNormalizedEnergyHtml = ...
      sprintf('%.1f&times;10<sup>%d</sup>', ...
              mostSignificantNormalizedEnergy / ...
              10^floor(log10(mostSignificantNormalizedEnergy)), ...
              floor(log10(mostSignificantNormalizedEnergy)));
  %mostSignificantAmplitudeHtml = ...
  %    sprintf('%.1f&times;10<sup>%d</sup>', ...
  %            mostSignificantAmplitude / ...
  %            10^floor(log10(mostSignificantAmplitude)), ...
  %            floor(log10(mostSignificantAmplitude)));
  mostSignificantSNRHtml = ...
      sprintf('%.1f', sqrt(2 * mostSignificantNormalizedEnergy));

  % begin html channel entry
  wlog(debugLevel, 2, '  writing html report...\n');
  fprintf(htmlFID, '<a name="%s"></a>\n', channelNameString);
  fprintf(htmlFID, '<h3>\n');
  fprintf(htmlFID, '<input id="input_%s" type="checkbox" checked\n', ...
          channelNameString);
  fprintf(htmlFID, '       onclick="toggleVisible(''%s'');" />\n', ...
          channelNameString);
  fprintf(htmlFID, ['<a href="https://cis.ligo.org/channel/' ...
                    'byname/%s">\n'], channelNameString);
  fprintf(htmlFID, '%s\n', channelNameString);
  fprintf(htmlFID, '</a>\n');
  fprintf(htmlFID, '</h3>\n');

  fprintf(htmlFID, '<div id="div_%s" style="display: block;">\n', ...
          channelNameString);

  fprintf(htmlFID, ['most significant tile:\n' ...
                    ' t = %s s,\n' ...
                    ' f = %s Hz,\n' ...
                    ' Q = %s,\n' ...
                    ' Z = %s,\n' ...
                    ' SNR = %s\n' ...
                    '<br />\n'], ...
          uniqueID, mostSignificantFrequencyHtml, ...
          mostSignificantQHtml, mostSignificantNormalizedEnergyHtml, ...
          mostSignificantSNRHtml);

  timeRangeString = sprintf('''%.2f'', ', plotTimeRanges);
  timeRangeString = timeRangeString(1 : end - 2);


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                       begin loop over time ranges                          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  toc;
  % begin loop over time ranges
  for plotTimeRange = plotTimeRanges,

    % report status
    wlog(debugLevel, 2, '  processing %.2f second time range...\n', plotTimeRange);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      plot whitened spectrogram                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % plot whitened spectrogram
    wlog(debugLevel, 2, '    plotting whitened spectrogram...\n');
    clf;
    wspectrogram(whitenedTransform, tiling, outputDirectory,uniqueID,startTime, centerTime, ...
                 plotTimeRange * [-1 +1] / 2, plotFrequencyRange, ...
                 mostSignificantQ, plotNormalizedEnergyRange, ...
                 plotHorizontalResolution);

    % print whitened spectrogram to file
    wlog(debugLevel, 2, '    printing whitened spectrogram...\n');
    figName = sprintf('%s_%s_%.2f_spectrogram_whitened', ...
                      uniqueID, channelNameString, plotTimeRange);
    figBasePath = [outputDirectory '/' figName];
    print(figureHandle,figBasePath,'-dpng');

    toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                      add images to html report                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % create html link
    wlog(debugLevel, 2, '    linking images...\n');
    fprintf(htmlFID, '<a id="a_%s_%s_%.2f"\n', ...
            uniqueID, channelNameString, plotTimeRange);
    fprintf(htmlFID, '   href="%s_%s_%.2f_spectrogram_whitened.png">\n', ...
            uniqueID, channelNameString, plotTimeRange);
    fprintf(htmlFID, '<img id="img_%s_%s_%.2f"\n', ...
            uniqueID, channelNameString, plotTimeRange);
    fprintf(htmlFID, '     src="%s_%s_%.2f_spectrogram_whitened.thumb.png"\n', ...
            uniqueID, channelNameString, plotTimeRange);
    fprintf(htmlFID, '     alt="%s_%s_%.2f" /></a>\n', ...
            uniqueID, channelNameString, plotTimeRange);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %                        end loop over time ranges                           %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % end loop time ranges
  end

  % end html table row
  fprintf(htmlFID, '<br />\n');
  fprintf(htmlFID, '</div>\n');
  fprintf(htmlFID, '\n');
   
  end % generateReport

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     end loop over configuration channels                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% end loop over configuration channels
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              close html report                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if generateReport,

% report status
wlog(debugLevel, 1, 'closing html report...\n');

% end any previously existing section
if inSectionFlag == 1,
  fprintf(htmlFID, '</div>\n');
  fprintf(htmlFID, '\n');
end

% end html
fprintf(htmlFID, '</div>\n');
fprintf(htmlFID, '\n');
fprintf(htmlFID, '<div class="footer">\n');
fprintf(htmlFID, 'Created by user %s on %s at %s<br />\n', ...
        getenv('USER'), datestr(clock, 29), datestr(clock, 13));
fprintf(htmlFID, '</div>\n');
fprintf(htmlFID, '\n');
fprintf(htmlFID, '</body>\n');
fprintf(htmlFID, '</html>\n');

% close html file
fclose(htmlFID);

end % generateReport

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          close text summary report                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'closing text summary...\n');

% close text summary file
fclose(textSummaryFID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           close xml summary report                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report status
wlog(debugLevel, 1, 'closing xml summary...\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     exit                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% report completion
wlog(debugLevel, 1, 'finished on %s at %s\n', ...
     datestr(clock, 29), datestr(clock, 13));

% close all figures
close all;

% return to calling function
return;
