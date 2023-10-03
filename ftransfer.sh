scp -r -i ~/.ssh/id_rsa ./*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/my_code/Liblearn
scp -r -i ~/.ssh/id_rsa ./models/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/my_code/Liblearn/models
scp -r -i ~/.ssh/id_rsa ./utils/*.py abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/my_code/Liblearn/utils
scp -r -i ~/.ssh/id_rsa ./custom_peft abhinav@anton.j"$1".rice.edu:/home/abhinav/prog_synth/my_code/Liblearn/custom_peft