#!/bin/sh
if [ ! -d images ]; then
    mkdir images
fi
if [ ! -d labels ]; then
    mkdir labels
fi
tar -C images -xvf stare-images.tar
tar -C labels -xvf labels-ah.tar 

gunzip -r labels
gunzip -r images
