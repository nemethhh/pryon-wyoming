#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
set -e

args=(
  --uri tcp://0.0.0.0:10400
  --models-dir "$(bashio::config 'models_dir')"
)
while IFS= read -r entry; do
  [ -n "$entry" ] && args+=(--threshold "$entry")
done < <(bashio::config 'thresholds')
if bashio::config.true 'debug_logging'; then
  args+=(--debug)
fi

/opt/venv/bin/pypryon-wyoming "${args[@]}" &
pid=$!

# announce to Home Assistant once it accepts connections
until (exec 3<>/dev/tcp/127.0.0.1/10400) 2>/dev/null; do
  kill -0 "$pid" 2>/dev/null || wait "$pid"
  sleep 0.2
done
bashio::discovery "wyoming" "{\"uri\": \"tcp://$(hostname):10400\"}" > /dev/null || bashio::log.warning "discovery failed"

wait "$pid"
