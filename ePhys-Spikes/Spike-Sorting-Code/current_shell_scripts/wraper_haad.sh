input_dirs=(\
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Kai/2021-12-17" \
# "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/2021-12-17" \
  )
ouput_dirs=(\
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/Kai/2021-12-17" \
# "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/2021-12-17" \
  )
filename_json="/pre_MS.json"
num_features_var=8
max_num_clips_for_pca_var=1000
cat mountainSort128_stroke_hyr2.sh > logs/mountainSort128_BC7.log
export ML_TEMPORARY_DIRECTORY=/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/ml_temp
ovr_start_stamp=$SECONDS
for i in "${!input_dirs[@]}"; do
  input_dir="${input_dirs[i]}"
  ouput_dir="${ouput_dirs[i]}"
  temp_path=$input_dir$filename_json
  samplerate=$(cat $temp_path | jq .SampleRate)
  geom_file="${ouput_dirs[i]}/geom.csv"
  num_features="$num_features_var"
  max_num_clips_for_pca="$max_num_clips_for_pca_var"
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
