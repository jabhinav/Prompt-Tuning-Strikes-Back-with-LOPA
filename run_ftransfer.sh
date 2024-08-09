#!/bin/bash

server_type="$1"
ip_address="$2"

if [ "$server_type" == "paperspace" ]; then
    # Paperspace commands
    echo "Running Paperspace commands..."

    # # Uncomment to transfer the scripts to paperspace
    scp -r -i ~/.ssh/id_rsa ./*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning
    scp -r -i ~/.ssh/id_rsa ./*.sh paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning
    scp -r -i ~/.ssh/id_rsa ./utils/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/utils
    scp -r -i ~/.ssh/id_rsa ./trainers/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/trainers
    scp -r -i ~/.ssh/id_rsa ./utils/cruxeval_tasks/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/utils/cruxeval_tasks

    # # Uncomment to update benchmarking scripts
    scp -r -i ~/.ssh/id_rsa ./custom_benchmark/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/custom_benchmark/

    # # Uncomment to update the peft scripts
    scp -r -i ~/.ssh/id_rsa ./custom_peft/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/custom_peft/
    scp -r -i ~/.ssh/id_rsa ./custom_peft/utils/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/custom_peft/utils/

    # # Uncomment to update cruxeval scripts
    scp -r -i ~/.ssh/id_rsa ./cruxeval/evaluation/*.py paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/cruxeval/evaluation/

    # # Uncomment to transfer the json files for deepspeed config. Generate yaml files for accelerator locally
#    scp -r -i ~/.ssh/id_rsa ./config_files/*.json paperspace@"$ip_address":/home/paperspace/prog_synth/LibraryLearning/config_files/

    #scp -r -i ~/.ssh/id_rsa ./custom_peft abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/

elif [ "$server_type" == "anton" ]; then
    # Anton commands
    echo "Sending files to Anton..."

    scp -r -i ~/.ssh/id_rsa ./*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/
    scp -r -i ~/.ssh/id_rsa ./*.sh abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/
    scp -r -i ~/.ssh/id_rsa ./utils/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/utils/
#    scp -r -i ~/.ssh/id_rsa ./utils/cruxeval_tasks/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/utils/cruxeval_tasks/
    scp -r -i ~/.ssh/id_rsa ./trainers/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/trainers/

    # # Uncomment to update benchmarking scripts
#    scp -r -i ~/.ssh/id_rsa ./custom_benchmark/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/custom_benchmark/

    # # Uncomment to update the peft scripts
#    scp -r -i ~/.ssh/id_rsa ./custom_peft/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/custom_peft/
#    scp -r -i ~/.ssh/id_rsa ./custom_peft/utils/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/custom_peft/utils/

    # # Uncomment to update
#    scp -r -i ~/.ssh/id_rsa ./cruxeval/evaluation/*.py abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/cruxeval/evaluation/

    # # Uncomment to transfer the yaml files for accelerator, json files for deepspeed config
    #scp -r -i ~/.ssh/id_rsa ./config_files/*.yaml abhinav@anton.j"$ip_address".rice.edu:/home/abhinav/prog_synth/Liblearn/config_files/

else
    echo "Invalid server_type. Please provide 'paperspace' or 'anton'."
    exit 1
fi
