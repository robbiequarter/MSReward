function s = freadstring(mf)
    str_size = flength(mf);
    s = char(fread(mf, str_size, 'char'));
end