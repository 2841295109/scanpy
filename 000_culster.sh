#!/usr/bin/env bash

#获取当前目录
current_dir=$(pwd)
echo "当前目录 $current_dir"

cd $current_dir
mkdir samples
mv *.tar samples/
cd samples

#创建子目录
for tar_file in *.tar; do
  echo " 处理：$tar_file"
  gsm_list=$(tar -tf "$tar_file" | grep -o 'GSM[0-9]*' | sort -u)
  tar -xf "$tar_file"
  for gsm in $gsm_list; do 
    echo "  创建文件夹: $gsm"
    mkdir -p "$gsm"
    for f in ${gsm}_*.tsv.gz ${gsm}_*.mtx.gz; do
      if [ -f "$f" ]; then
        mv -v "$f" "$gsm/"
      fi
    done
  done
done


  

