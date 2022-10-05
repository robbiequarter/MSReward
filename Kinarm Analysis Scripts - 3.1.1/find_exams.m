function exams = find_exams( root, bRecurse )

    paths = dir( root );
    exams = {};
    
    for i=1:length(paths)
        
        if paths(i).name(1) == '.'
            continue            
        elseif paths(i).isdir && bRecurse
            sub_exams = find_exams([root '/' paths(i).name]);
            for n=1:length(sub_exams)
                exams{end+1} = sub_exams{n};
            end
        elseif strfind(paths(i).name, '.zip')
            path = [root '/' paths(i).name];
            
            if isExam(path)
                exams{end+1} = path;
            end
        end                
    end
end

function bIsExam = isExam(path)
    zf = java.util.zip.ZipFile(java.io.File(path), java.util.zip.ZipFile.OPEN_READ);
    entries = zf.entries();
    bIsExam = false;
    
    while (entries.hasMoreElements())
        ze = entries.nextElement();     
        if strfind(ze.getName(), 'raw/') == 1
            bIsExam = true;
            break;
        end
    end
    
    zf.close();
end

