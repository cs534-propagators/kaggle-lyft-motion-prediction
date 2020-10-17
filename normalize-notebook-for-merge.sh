#!/usr/bin/bash

# environment variables
source ~/.bashrc

RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'

# helper functions

printColor(){
  COLOR=$1
  MESSAGE=$2
  printf "${COLOR}${MESSAGE}${NC}\n"
}

printError(){
  MESSAGE="$1"
  printColor $RED "Error: $MESSAGE"
  exit -1
}

printSuccess(){
  MESSAGE="$1"
  printColor $GREEN $MESSAGE
}

# start of script
FILE_INPUT="$1"

# check input file argument
if [ -z $FILE_INPUT ]
then
  printError "Must specify input file as first argument!"
fi

# check that input file exists
if [ ! -f $FILE_INPUT ]
then
  printError "Input file \"$FILE_INPUT\" must exist!"
fi

FILE_BASE=$(echo $FILE_INPUT | sed -e 's/\.ipynb//')
FILE_OUTPUT="$FILE_BASE.py"
FILE_TMP="$FILE_BASE.py.tmp"

# convert notebook to python script
jupyter-nbconvert --to script $FILE_INPUT

# check that output file exists
if [ ! -f $FILE_OUTPUT ]
then
  printError "Output file \"$FILE_OUTPUT\" must exist!"
fi

# normalize script by replacing cell numbers
FIND='# *In\[ *[0-9]* *\] *: *'
REPLACE='# In[]:'
cat $FILE_OUTPUT | sed -e "s/$FIND/$REPLACE/" >$FILE_TMP

# check that tmp file exists
if [ ! -f $FILE_TMP ]
then
  printError "Temporary file \"$FILE_TMP\" must exist!"
fi

mv $FILE_TMP $FILE_OUTPUT
printSuccess "SUCCESS!"

