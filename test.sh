#!/usr/bin/env bash
if [[ ! -d .v ]]; then
	python3 -m venv .v
	source .v/bin/activate
	python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
fi
source .v/bin/activate
BIRDS="$(find images/birds/ -type f|tr '\n' ',')"

birds(){
	cmd="time python3 ./test.py \
		--model models/birds.tflite \
		--labels labels/birds.txt \
		--input $BIRDS"
	echo -e "$cmd" >&2
	eval "$cmd"
}

main(){
	birds
}

main
