input_dirs=(\
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/Yifu/2021-08-28-aged/2021-12-15"\
  )
ouput_dirs=(\
"/media/luanlab/DATA/SpikeSorting/SpikeSortOut/2021-08-28-aged/2021-12-15"\
 )
 #/media/luanlab/DATA/SpikeSorting/RawData/2021-09-04B-aged/2022-01-01/2022-01-01_moving


num_features_var=(8) ##8 pca component
max_num_clips_for_pca_var=(1000) #1000 
samplerate=30000 # Sampling rate
#cat mountainSort128_stroke_hyr2.sh > logs/mountainSort128_BC7.log
export ML_TEMPORARY_DIRECTORY=/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/ml_temp #temporary files
ovr_start_stamp=$SECONDS
for i in "${!input_dirs[@]}"; do
  input_dir="${input_dirs[i]}"
  ouput_dir="${ouput_dirs[i]}"
  samplerate=$samplerate
  geom_file="${ouput_dirs[i]}/geom.csv"
  num_features="${num_features_var[i]}"
  max_num_clips_for_pca="${max_num_clips_for_pca_var[i]}"
  echo ---------------------------------------
  echo "Executing command:" 
  echo ./mountainSort128_stroke_hyr2.sh $input_dir $ouput_dir $samplerate $geom_file $num_features $max_num_clips_for_pca
  echo ---------------------------------------
  session_start_stamp=$SECONDS
  ./mountainSort128_stroke_hyr2.sh $input_dir $ouput_dir $samplerate $geom_file $num_features $max_num_clips_for_pca
  echo "Session finished. Deleting temp files..." 
  rm -rf $ML_TEMPORARY_DIRECTORY
  echo "Session finished in " $(( SECONDS - session_start_stamp )) " seconds."
done
echo "All sessions done in " $(( SECONDS - ovr_start_stamp )) " seconds."
