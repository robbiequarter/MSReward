classdef MemoryFile < handle
    %MEMORYFILE A file that is read into memory and then read from there.
    
    properties
        fid
        requiresSwap
    end
    
    properties(Constant)
        swapRequiredLE = java.nio.ByteOrder.LITTLE_ENDIAN ~= java.nio.ByteOrder.nativeOrder();
        swapRequiredBE = java.nio.ByteOrder.BIG_ENDIAN ~= java.nio.ByteOrder.nativeOrder();
    end
    
    methods
        function obj = MemoryFile(byteData, isDataLE)                      
            obj.fid = mfile(typecast(byteData, 'uint8'));
            fseek(obj.fid, 0, 'bof');
            
            if isDataLE
                obj.requiresSwap = obj.swapRequiredLE;
            else
                obj.requiresSwap = obj.swapRequiredBE;
            end
        end
        

        function ret = readInt(obj)
            ret = fread_int(obj.fid, 1, obj.requiresSwap);
        end

        function ret = readInts(obj, count)
            if count == 0
                ret = [];
            else
                ret = fread_int(obj.fid, count, obj.requiresSwap);
            end
        end
        
        function ret = readFloat(obj)
            ret = obj.readFloats(1);
        end
        
        function ret = readString(obj)
            slen = fread_int(obj.fid, 1, obj.requiresSwap);
            if slen == 0
                ret = char([]);
            else
                bytes = int8(fread(obj.fid, slen * 2, 'int8'));
                ret = native2unicode(bytes, 'UTF-16LE');
            end
        end
        
        function ret = readFloats(obj, count)
            if count == 0
                ret = [];
            else                
%                 tic
                if obj.requiresSwap
                    % KAS-51 - The "*float32" forces the function to return singles.
                    % this makes it possible for the byte reordering to
                    % work properly without accidentally finding values
                    % that are NaN
%                     ret = fread_int_swap(obj.fid, count);
%                     ret  = double(typecast(ret, 'single'));
                    ret = fread(obj.fid, count, '*float32');
                    ret = double(typecast(swapbytes(typecast(ret, 'int32')), 'single'));
                else
                    ret = fread(obj.fid, count, 'float32');
                end
%                 toc
            end
        end
        
        function ret=available(obj)
            ret = ftell(obj.fid) < flength(obj.fid);
        end
        
        function close(obj)
        end
    end
    
end

