input_dirs=(\
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/NVC/B-BC8/3-30-22/ePhys" \
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/NVC/B-BC8/3-31-22/ePhys" \
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/converted/data_mda/NVC/B-BC8/4-04-22/ePhys" \
  )
ouput_dirs=(\
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/B-BC8/3-30-22/ePhys" \
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/B-BC8/3-31-22/ePhys" \
"/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out/NVC/B-BC8/4-04-22/ePhys" \
  )
samplerate=25000
cat mountainSort128_stroke_jz103.sh > logs/mountainSort128_B-BC8.log
export ML_TEMPORARY_DIRECTORY=/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/ml_temp
ovr_start_stamp=$SECONDS
for i in "${!input_dirs[@]}"; do
  input_dir="${input_dirs[i]}"
  ouput_dir="${ouput_dirs[i]}"
  samplerate=$samplerate
  geom_file="${ouput_dirs[i]}/geom.csv"
  echo ---------------------------------------
  echo "Executing command:" 
  echo ./mountainSort128_stroke_jz103.sh $input_dir $ouput_dir $samplerate $geom_file
  echo ---------------------------------------
  session_start_stamp=$SECONDS
  ./mountainSort128_stroke_jz103.sh $input_dir $ouput_dir $samplerate $geom_file
  echo "Session finished. Deleting temp files..."
  rm -rf $ML_TEMPORARY_DIRECTORY
  echo "Session finished in " $(( SECONDS - session_start_stamp )) " seconds."
done
echo "All sessions done in " $(( SECONDS - ovr_start_stamp )) " seconds."
