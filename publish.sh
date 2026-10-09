#!/usr/bin/env bash
# Upload today's output to designer2.org over FTPS (credentials from ~/.netrc).
# latest.json goes last, so the app never sees a latest.json pointing to a missing file.
#
#   ./publish.sh       v1: out/    -> data/     (HR+SI, the 0.4.x apps)
#   ./publish.sh v2    v2: out/v2/ -> data/v2/  (9 countries + rain layer, 0.5.0+ apps)
set -euo pipefail
cd "$(dirname "$0")/out"

REMOTE=designer2/gljivar/data
if [ "${1:-}" = "v2" ]; then
  cd v2
  REMOTE=$REMOTE/v2
fi

FILE=$(python3 -c "import json; print(json.load(open('latest.json'))['file'])")
# v2 also has the rain layer
RAIN=$(python3 -c "import json; print(json.load(open('latest.json')).get('rain', {}).get('file', ''))")
DATA="$FILE $RAIN"
PUTS=""
for f in $DATA grid.json; do PUTS="$PUTS put $f;"; done

# CI passes FTP_USER / FTP_PASS; locally lftp falls back to ~/.netrc.
AUTH=()
if [ -n "${FTP_USER:-}" ]; then
  export LFTP_PASSWORD="$FTP_PASS"
  AUTH=(-u "$FTP_USER" --env-password)
fi

lftp ${AUTH[@]+"${AUTH[@]}"} ftp.designer2.org -e "set ftp:ssl-allow yes; set ssl:verify-certificate no; set net:max-retries 3; \
  cd public_html; mkdir -p $REMOTE; cd $REMOTE; \
  $PUTS put latest.json; \
  chmod 755 .; chmod 644 $DATA grid.json latest.json; quit"

URL=https://designer2.org/$REMOTE
curl -sf "$URL/latest.json" | grep -q "\"file\": \"$FILE\"" || { echo "latest.json on server does not point to $FILE"; exit 1; }
for f in $DATA; do
  test "$(curl -sf "$URL/$f" | wc -c | tr -d ' ')" = "$(wc -c < "$f" | tr -d ' ')" || { echo "size mismatch for $f"; exit 1; }
done
echo "published $URL/$FILE"
