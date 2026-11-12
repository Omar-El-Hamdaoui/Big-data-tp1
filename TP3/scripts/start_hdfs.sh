#!/usr/bin/env bash
set -euo pipefail
# Réutilise l'installation existante. Ne jamais exécuter « namenode -format » ici.
HADOOP_BIN="${HADOOP_HOME:-/home/omar/hadoop}/bin/hdfs"
if "$HADOOP_BIN" dfs -ls / >/dev/null 2>&1; then
  echo 'HDFS est déjà accessible.'
else
  "$HADOOP_BIN" --daemon start namenode
  "$HADOOP_BIN" --daemon start datanode
  "$HADOOP_BIN" dfsadmin -safemode wait
fi
"$HADOOP_BIN" dfsadmin -report
