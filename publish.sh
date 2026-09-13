#!/usr/bin/env bash
# Upload today's output to designer2.org over FTPS (credentials from ~/.netrc).
# latest.json goes last, so the app never sees a latest.json pointing to a missing file.
set -euo pipefail
cd "$(dirname "$0")/out"

FILE=$(python3 -c "import json; print(json.load(open('latest.json'))['file'])")
REMOTE=designer2/gljivar/data

lftp ftp.designer2.org -e "set ftp:ssl-allow yes; set ssl:verify-certificate no; set net:max-retries 3; \
  cd public_html; mkdir -p $REMOTE; cd $REMOTE; \
  put $FILE; put grid.json; put latest.json; \
  chmod 755 .; chmod 644 $FILE grid.json latest.json; quit"

URL=https://designer2.org/$REMOTE
curl -sf "$URL/latest.json" | grep -q "\"file\": \"$FILE\"" || { echo "latest.json on server does not point to $FILE"; exit 1; }
test "$(curl -sf "$URL/$FILE" | wc -c | tr -d ' ')" = "$(wc -c < "$FILE" | tr -d ' ')" || { echo "size mismatch for $FILE"; exit 1; }
echo "published $URL/$FILE"
