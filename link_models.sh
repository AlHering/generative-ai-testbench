#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ln -sf $1/MODELS/* $SCRIPT_DIR/machine_learning_models/MODELS/
ln -sf $1/LORAS/* $SCRIPT_DIR/machine_learning_models/LORAS/