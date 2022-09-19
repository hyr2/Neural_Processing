function write_sc(sc,fname)
%write_sc Writes array to speckle contrast file

  [im_h,im_w,Nimg] = size(sc);

  fileID = fopen(fname,'wb');

  % Write header
  fwrite(fileID, im_w, 'ushort');
  fwrite(fileID, im_h, 'ushort');
  fwrite(fileID, Nimg, 'ushort');

  % Write data
  fwrite(fileID, reshape(sc,[],1), 'float');

  fclose(fileID);

end