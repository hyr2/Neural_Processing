animal_folder=/media/luanlab/Data_Processing/Jim-Zhang/SpinalCordSpikeSort/raw_data/Prius_11182021_IrOx_4refOut
output_folder=/media/luanlab/Data_Processing/Jim-Zhang/SpinalCordSpikeSort/msort_outputs/Prius_11182021_IrOx_4refOut
#geom_file=map_corolla24ch.csv
geom_file=map.csv
export ML_TEMPORARY_DIRECTORY=/media/luanlab/Data_Processing/Jim-Zhang/SpinalCordSpikeSort/ml_temp
if [ -d $output_folder ] 
then
    echo "Directory \"$output_folder\" exists." 
else
    echo "Creating directory: \"$output_folder\""
    mkdir -p $output_folder
fi
samplerate=30000
ovr_start_stamp=$SECONDS
sessions=$(ls $animal_folder)
#sessions=("nacho_awake_210907_205310")
for session_folder in ${sessions[*]}
  do
    path=$animal_folder/$session_folder
    outpath=$output_folder/$session_folder
    if [ -d $outpath ] 
    then
        echo "Directory \"$outpath\" exists." 
    else
        echo "Creating directory: \"$outpath\""
        mkdir $outpath
    fi
    echo ---------------------------------
    echo Executing following command:
    echo ./mountainSort32_spinalCord_jz103.sh $path $outpath $samplerate $geom_file
    session_start_stamp=$SECONDS
    ./mountainSort32_spinalCord_jz103.sh $path $outpath $samplerate $geom_file
    echo "Session finished. Deleting temp files..."
    rm -f $ML_TEMPORARY_DIRECTORY/*
    echo "Session finished in " $(( SECONDS - session_start_stamp )) " seconds."
  done
echo "All sessions done in " $(( SECONDS - ovr_start_stamp )) " seconds."
