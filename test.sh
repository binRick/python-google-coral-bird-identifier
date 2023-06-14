#!/usr/bin/env bash
if [[ ! -d .v ]]; then
	python3 -m venv .v
	source .v/bin/activate
	python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
fi
source .v/bin/activate
BIRDS="$(find images/birds/ -type f|tr '\n' ',')"
PRODUCTS="$(find images/products/ -type f|tr '\n' ',')"
GENDERS="$(find images/genders/ -type f|tr '\n' ',')"

products(){
	cmd="time python3 ./test.py \
		--model models/products.tflite \
		--labels labels/products.csv \
		--input $PRODUCTS"
	echo -e "$cmd" >&2
	eval "$cmd"
}
genders(){
	cmd="time python3 ./test.py \
		--model models/genders.tflite \
		--labels labels/genders.txt \
		--input $GENDERS"
	echo -e "$cmd" >&2
	eval "$cmd"
}
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
	products
	#genders
}

main
