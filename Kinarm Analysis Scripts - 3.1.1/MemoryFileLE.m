classdef MemoryFileLE < handle
    %MEMORYFILE A file that is read into memory and read from there.
    %Assumes data is stored in Little Endian format.
    
    properties
        fid
        requiresSwap
    end
    
    methods
        function obj = MemoryFileLE(byteData)                      
            obj.fid = mfile(typecast(byteData.array(), 'uint8'));
            fseek(obj.fid, 0, 'bof');
            obj.requiresSwap = java.nio.ByteOrder.LITTLE_ENDIAN ~= java.nio.ByteOrder.nativeOrder();
        end
        

        function ret = readInt(obj)
            ret = fread(obj.fid, 1, 'int32');
            if obj.requiresSwap
                ret = swapbytes(int32(ret));
            end
        end
        
        function ret = readString(obj)
            slen = obj.readInt();
            bytes = int8(fread(obj.fid, slen * 2, 'int8'));
            jret = java.lang.String(bytes, java.nio.charset.StandardCharsets.UTF_16LE);
            ret = char(jret);
        end
        
        function ret = readFloats(obj, count)
            ret = fread(obj.fid, count, 'float32');
            if obj.requiresSwap
                ret = single(ret);
                ret = double(typecast(swapbytes(typecast(ret, 'int32')), 'single'));
            end
        end
         
        function close(obj)
        end
    end
    
end

