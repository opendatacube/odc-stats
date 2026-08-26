#!/bin/bash

# $1: path to find the files
# $2: file name pattern used to find the files
# $3...: extra args passed through to compare_data.py, e.g. --max-diff-pct 0.5


if [[ $1 == "" ]]
then FILE_PATH="./" 
else FILE_PATH=$1
fi

if [[ $2 == "" ]]
then SEARCH_PATTERN="*" 
else SEARCH_PATTERN=$2
fi

S3_PREFIX="/vsis3/dea-public-data-dev/stats-golden-files/"
for f in $(find $FILE_PATH -name "$SEARCH_PATTERN")
do  
	if [[ $f == /* ]]
	then g_name=$(echo $f | cut -d'/' -f3-)
	else g_name=$(echo $f | cut -d'/' -f2-)
	fi
	python3 "$(dirname "$0")/compare_data.py" $S3_PREFIX$g_name $f "${@:3}"
	if [[ $? -ge 1 ]]
	then 
		echo "ERROR: results different from the golden file $S3_PREFIX$g_name"
		exit 1
	else
		echo "$f pass test "
	fi
done
