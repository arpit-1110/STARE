#!/bin/sh

tar -C images -xvf stare-images.tar
tar -C labels -xvf labels-ah.tar 

gunzip -r labels
gunzip -r images
