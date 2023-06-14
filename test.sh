#!/usr/bin/env bash
if [[ ! -d .v ]]; then
	python3 -m venv .v
	source .v/bin/activate
	python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
fi
source .v/bin/activate
IMAGES="\
	parrot.jpg \
	tufted_flycatcher.jpg \
"
for I in $IMAGES; do
 time python3 ./test.py \
	--model mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
	--labels bird_labels.txt \
	--input images/$I
done
