#!/usr/bin/env bash

#获取当前目录
current_dir=$(pwd)
echo "当前目录 $current_dir"

#建目录
cd $current_dir
mkdir samples
mkdir figure

#处理tar文件
mv *.tar samples/
cd samples
for tar_file in *.tar; do
	echo"处理：$tar_file"
	
	# 获取所有 GSM 前缀（去重）
    gsm_list=$(tar -tf "$tar_file" | grep -o 'GSM[0-9]*' | sort -u)
    
    # 为每个 GSM 创建文件夹并提取对应文件
    for gsm in $gsm_list; do
        echo "  创建文件夹: $gsm"
        mkdir -p "$gsm"
        
        # 提取该 GSM 对应的所有文件 先目录后通配
		tar -xf "$tar_file" -C "$gsm" --wildcards "*${gsm}*" 
		cd $gsm
		mv -v *barcodes*.tsv.gz barcodes.tsv.gz
		mv -v *features*.tsv.gz features.tsv.gz
		mv -v *matrix*.mtx.gz matrix.mtx.gz
		cd ..
		echo "$dir 重命名已完成。"
        echo "  完成: $gsm"
    done
done
echo "所有文件分类完成！"
