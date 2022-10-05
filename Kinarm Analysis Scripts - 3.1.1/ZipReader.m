classdef ZipReader < handle
    %ZIP_READER This class is set up to use some standard java calls to
    %read the contents of a zip file into memory as and array of uint8. The
    %data can the be read using a memory file using standard fread, ftell
    %etc functions.
    
    properties
        bClasicFormat
        file_name
        zip_contents
        folder_map
        openFS
        fsRoot
    end
    
    methods
        function obj = ZipReader(fname)
            % if we ever decide to use custom Java code then this is how to
            % load it.
%             path = which('zip_load.m');
%             path = path(1:length(path) - length('zip_load.m'));
%             javaaddpath([path 'kinarm.jar']);                       
            obj.bClasicFormat = ~isempty(strfind(fname, '.zip'));
            obj.file_name = fname;
            [obj.openFS, obj.fsRoot] = openZipFileSystem(fname);
            zc = obj.dir();          
            zip_map = containers.Map();
            
            % builds up a map of folder names to folder contents
            for n=1:length(zc)
                fname = zc{n};              
                [folder, name, ~] = fileparts(fname);
                if ~isempty(folder)                
                    if ~isKey(zip_map, folder)
                        zip_map(folder) = {};
                    end
                    if ~isempty(name)
                        arr = zip_map(folder);
                        arr{end+1} = fname;
                        zip_map(folder) = arr;
                    end
                end
            end
            obj.folder_map = zip_map;
            obj.zip_contents = zc;
        end
        
        function zip_contents=dir(obj)
            
            if isempty(obj.zip_contents)                
                zf = obj.openZF();
                zip_contents{zf.size()} = [];
                entries = zf.entries();
                i = 1;
                
                while (entries.hasMoreElements())  
                    zip_contents{i} = entries.nextElement().getName().toCharArray';
                    i = i + 1;
                end

                zf.close();
                obj.zip_contents = zip_contents;               
                return                
            end
            zip_contents = obj.zip_contents;
        end

        function trial_names=listTrials(obj)   
            % Find the names of all trial in the exam
            trial_names = {};
            
            if obj.bClasicFormat
                % Trials are in the raw folder and end in .c3d
                rawData = obj.folder_map('raw');
                for i=1:length(rawData)
                    name = rawData{i};
                    if ~isempty(strfind(name, '.c3d')) && isempty(strfind(name, 'common.c3d'))
                        trial_names{end+1} = name;
                    end
                end
            else
                % trials are folder in the raw folder.
                names = keys(obj.folder_map);
                for i=1:length(names)
                    name = names{i};
                    if obj.startsWith(name, 'raw') && isempty(strfind(name, 'common'))
                        trial_names{end+1} = char(name);
                    end
                end
            end
        end
        
        function bSuccess=startsWith(obj, val, find)
            bSuccess = false;
            
            if isempty(strfind(val, find))
                return
            end
            bSuccess = strfind(val, find) == 1 && length(val) ~= length(find);               
        end

        function fnames=listAnalysis(obj)                      
            % Find the names of all analysis files in the exam
            fnames = {};
            
            if obj.bClasicFormat
                if isKey(obj.folder_map, 'analysis')
                    rawData = obj.folder_map('analysis');
                    for i=1:length(rawData)
                        name = rawData{i};
                        if ~isempty(strfind(name, '.c3d'))
                            fnames{end+1} = name;
                        end
                    end
                end
            else
                names = keys(obj.folder_map);
                for i=1:length(names)
                    name = names{i};
                    if ~isempty(strfind(name, 'analysis') == 1) && ~strcmp(name, 'analysis')
                        fnames{end+1} = char(name);
                    end
                end
            end
        end
        
        function bHas = containsFile(obj, name)
            bHas = ~isempty(find(strcmp(obj.zip_contents, name),1));
        end

        function bHas = containsFolder(obj, name)
            bHas = isKey(obj.folder_map, name);
        end

        % Construct a fid that is a memory based file for reading an entry
        % in a zip file
        function fid=fopen(obj, path)          
            bb = obj.readIntobyteBuffer(path);
            fid = mfile(typecast(bb, 'uint8'));
            fseek(fid, 0, 'bof');
        end

 
        function nameMap=readFolder(obj, path)
            % For the new data structure (.kinarm files), this reads in all 
            % of the data for a single trial at once. This is much faster
            % than reading the zip entries individually.
            % This method returns a Map object that is zip entry name ->
            % file reading struct.
            [zipFileSystem, root, doClose] = obj.getOpenFileSystem();
            
            entrynames = obj.folder_map(path);
            names{length(entrynames)} = [];
            fids{length(entrynames)} = [];
            
            for i=1:length(entrynames)
                names{i} = entrynames{i};                
                pathInZip = root.resolve(entrynames{i});
                byteData = java.nio.file.Files.readAllBytes(pathInZip);
                % if the version number is in the first byte then we know
                % the data is Little Endian. No version that went to
                % customers was ever Big Endian.
                bLE = byteData(1) ~= 0; 
                %bLE = ~isempty(strfind(names{i}, '.kinematics')) || ~isempty(strfind(names{i}, '.position'));                    
                fids{i} = make_file_struct(byteData, bLE);
            end
            
            nameMap = containers.Map(names, fids);  
            if doClose
                zipFileSystem.close();
            end
        end
        
        function close(obj)
            % Close the open FileSystem object if it is still open.
           if ~isempty(obj.openFS)
               obj.openFS.close(); 
               obj.openFS = [];
           end
        end
    end
    
    methods (Access=private)
        
        function zf=openZF(obj)
            zf = java.util.zip.ZipFile(java.io.File(obj.file_name), java.util.zip.ZipFile.OPEN_READ);
        end

        function [fs, root, doClose] = getOpenFileSystem(obj)
            if isempty(obj.openFS)
                [fs, root] = openZipFileSystem(obj.file_name);
                doClose = 1;
            else
                fs = obj.openFS;
                root = obj.fsRoot;
                doClose = 0;
            end
        end
        
        function bb=readIntobyteBuffer(obj, path)
            [zipFileSystem, root, doClose] = obj.getOpenFileSystem();

            p = root.resolve(path);
            bb = java.nio.file.Files.readAllBytes(p);
            
            if doClose
                zipFileSystem.close();
            end
        end
    end
    
end

function [openFS, fsRoot] = openZipFileSystem(zip_name)
    % The fastest way to access a zip file from MATLAB is to mount it using 
    % a FileSystem object in Java and then use Java's nio methods to read
    % the files.
    % This returns the open file system object and the path to the root of
    % the file system. The path is required because it is the most efficent
    % way to find the files in the zip (using Path.resolve())
    
    % When the path to the zip is not an absolute path then MATLAB has
    % trouble resolving it properly. To help MATLAB, I prepend the current
    % working folder to any path that is relative
    ff = java.io.File(zip_name);
    if ~ff.isAbsolute()
        ff = java.io.File(pwd, zip_name);
    end

    % Normally we just try to mount the zip (newFileSystem), but if it was
    % not properly closed then an exception is thrown and we can instead to
    % to get the already mounted zip.
    try
        openFS = java.nio.file.FileSystems.newFileSystem(java.net.URI.create(['jar:' char(ff.toURI().toString())]), java.util.HashMap());
    catch ME
        openFS = java.nio.file.FileSystems.getFileSystem(java.net.URI.create(['jar:' char(ff.toURI().toString())]));
    end
    
    % The java Path object for the root folder gives us the most efficient
    % way to access files in the zip.
    fsRoot = openFS.getRootDirectories().iterator().next();
end

function fstruct = make_file_struct( byteData, isDataLE )
    % This function takes an array of data and returns a structure with a
    % mfile object (an in memory file essentially).
    % The isDataLE is 1 if the data in the file is Little Endian, 0 if the
    % data is Big Endian.
    persistent swapRequiredBE swapRequiredLE
    
    if isempty(swapRequiredBE)
        swapRequiredLE = java.nio.ByteOrder.LITTLE_ENDIAN ~= java.nio.ByteOrder.nativeOrder();
        swapRequiredBE = java.nio.ByteOrder.BIG_ENDIAN ~= java.nio.ByteOrder.nativeOrder();
    end

    fid = mfile(typecast(byteData, 'uint8'));
    fseek(fid, 0, 'bof');

    % determine if the data in the file needs to have its byte order
    % swapped during reading
    if isDataLE
        requiresSwap = swapRequiredLE;
    else
        requiresSwap = swapRequiredBE;
    end

    fstruct = struct('fid', fid, 'requiresSwap', requiresSwap);
end
