model_name=extreme
for d in ${model_name}*
do
    count_pth=`ls ${d}/*pth 2> /dev/null | wc -l`
    if [[ $count_pth == 0 ]]
    then
        ls -d $d
        rm -r $d
    fi
done
