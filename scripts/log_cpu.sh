while true; do ps -p $1 -o %C | sed -n '2p' >> logs/cpu_log.csv; sleep 0.1; done
