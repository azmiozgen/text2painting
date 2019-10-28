SCRIPT_IDENTIFICATION=$(dirname "${0}")/pred_script.py

PROCESS_DIR=${1}
PYTHON_PATH=${2}
USEGPU=${3}
INPUT_IMAGE=${4:-"Image.png"}

if $usegpu;
then
  $PYTHON_PATH $SCRIPT_IDENTIFICATION --dir $PROCESS_DIR --input $INPUT_IMAGE --usegpu
else
  $PYTHON_PATH $SCRIPT_IDENTIFICATION --dir $PROCESS_DIR --input $INPUT_IMAGE 
fi
