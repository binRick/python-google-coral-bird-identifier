#!/usr/bin/env bash
if [[ ! -d .v ]]; then
	python3 -m venv .v
	source .v/bin/activate
	pip install numpy Pillow pycoral
	#python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
fi
source .v/bin/activate
BIRDS="$(find images/birds/ -type f|tr '\n' ',')"
PRODUCTS="$(find images/products/ -type f|tr '\n' ',')"
GENDERS="$(find images/genders/ -type f|tr '\n' ',')"
BANANAS="$(find images/bananas/ -type f|tr '\n' ',')"

bananas(){
	cmd="time python3 ./test.py \
		--model models/mobilenet_v1_0.75_192_quant_edgetpu.tflite \
		--labels labels/mobilenet_v1_0.75_192_quant_edgetpu.txt \
		--input $BANANAS"
	echo -e "$cmd" >&2
	eval "$cmd"
}
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
	#birds
	#products
	bananas
	#genders
}

main
