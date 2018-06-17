#!/bin/bash
image_name=image.img
recipe_file=Singularity
usage() { echo "Usage: $0 [-r <recipe file path> ] [-n <image file path>]" 1>&2; exit 1; }
while getopts "hn:r:" arg; do
  case $arg in
    h)
      usage
      ;;
    r)
      recipe_file=$OPTARG
      ;;
    n)
      image_name=$OPTARG
      ;;
  esac
done
echo Singularity Build. Recipe: $recipe_file, Image: $image_name
sudo /opt/singularity/bin/singularity build  $image_name $recipe_file
