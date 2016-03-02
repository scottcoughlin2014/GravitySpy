function structure = wcopyevents(structure,indices)
% WCOPYEVENTS copy event rows from an Omega event structure
%
% structure = WCOPYEVENTS(structure,indices)

% apply default arguments
if (nargin < 2),
  indices = 'all';
end

% get list of available field names
fieldNames = fieldnames(structure);

% remove id and overflowFlag fields from list of field names
fieldNames = fieldNames(~strcmp(fieldNames, 'id') & ...
                        ~strcmp(fieldNames, 'overflowFlag') & ...
                        ~strcmp(fieldNames, 'channelName'));

% number of available field names
numberOfFields = length(fieldNames);

if strcmp(indices,'all'),
  indices = 1:length(structure.(fieldNames{1}));
end

% extract most significant signal events
for fieldNumber = 1 : numberOfFields,
    structure.(fieldNames{fieldNumber}) = ...
        structure.(fieldNames{fieldNumber})(indices);
end
