# GATE

Getting Started

1) Install required packages via
pip3 install -r requirements.txt
2) Download Elmo3 from https://tfhub.dev/google/elmo/3 and put it in under GATE folder
 --GATE
  --elmo_3
    --variables
    --saved_model.pb
    --tfhub_module.pb
  --creator
  --etc
3) You can change the parameters from config.txt file
4) Finally, to execute

python3 main.py config.txt
