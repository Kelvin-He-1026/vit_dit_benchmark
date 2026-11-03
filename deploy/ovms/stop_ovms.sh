#!/usr/bin/env bash
# Remove every container start_ovms.sh started.
ids=$(docker ps -aq --filter label=ovms-bench=1)
[[ -n $ids ]] && docker rm -f $ids >/dev/null && echo "removed $(echo $ids | wc -w) container(s)" || echo "none running"
