nohup spark-submit \
--num-executors 4 \
--executor-cores 5 \
--executor-memory 50G \
--driver-memory 20G \
./clean_synonyms.py > ./logs/spark-output.synonyms.txt 2> ./logs/spark-error.synonyms.txt &