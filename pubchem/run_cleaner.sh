nohup spark-submit \
--num-executors 4 \
--executor-cores 5 \
--executor-memory 50G \
--driver-memory 20G \
--driver-cores 5 \
./clean_summaries.py > ./logs/spark-output.clean_summaries.txt 2> ./logs/spark-error.clean_summaries.txt &