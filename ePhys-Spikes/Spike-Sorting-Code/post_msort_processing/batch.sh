#!/bin/bash
animalfoldername="/media/hanlin/Liuyang_10T_backup/jiaaoZ/128ch/spikeSorting128chHaad/spikesort_out221206/"
sessiondirs=(\
"11-11-2021"\
  )
for session in $sessiondirs; do
  sessionfoldername=$animalfoldername$session
  echo $sessionfoldername
  python discard_noise_and_viz_HR.py $sessionfoldername
done
