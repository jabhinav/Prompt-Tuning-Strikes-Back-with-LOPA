scp -r -i ~/.ssh/id_rsa ./*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/Liblearn/
scp -r -i ~/.ssh/id_rsa ./utils/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/Liblearn/utils/
scp -r -i ~/.ssh/id_rsa ./custom_peft/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/Liblearn/custom_peft/
scp -r -i ~/.ssh/id_rsa ./custom_peft/utils/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/Liblearn/custom_peft/utils/

#scp -r -i ~/.ssh/id_rsa ./*.py paperspace@209.51.170.18:/home/paperspace/LibraryLearning
#scp -r -i ~/.ssh/id_rsa ./models/*.py paperspace@209.51.170.18:/home/paperspace/LibraryLearning/models
#scp -r -i ~/.ssh/id_rsa ./utils/*.py paperspace@209.51.170.18:/home/paperspace/LibraryLearning/utils
#scp -r -i ~/.ssh/id_rsa ./custom_peft paperspace@184.105.5.246:/home/paperspace/LibraryLearning/custom_peft
