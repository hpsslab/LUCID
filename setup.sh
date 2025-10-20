#!/bin/bash

# This was 100% vibe coded (was not trying to relearn bash after all this time). Thx, Cursor :D

EXPECTED_LINES=518
PG_FILE=./venv/lib/python3.12/site-packages/llama_index/core/indices/property_graph/base.py

# Make sure we can't run this multiple times. That would be bad.
if [ "$(wc -l < "$PG_FILE")" -eq "$EXPECTED_LINES" ]; then
  echo "Error: this should only be run once."
  exit 0
fi

# Asynchronously insert things into the graph (instead of assuming a non asynchronous environment)
sed -i '224s/^\(\s*\)/\1await /' ./venv/lib/python3.12/site-packages/llama_index/core/indices/base.py

# Make the insert function itself asynchronous, have it call our copy of the function
sed -i '401s/^\(\s*\)/\1await /' $PG_FILE
sed -i '399s/^\(\s*\)/\1async /' $PG_FILE
sed -i '401s/insert/ainsert/' $PG_FILE

# Easiest way to do this, I figure, is just write a duplicate of the problematic function. 
# Keep one for the base purpose, revise one for our purposes.
gawk -i inplace -v N=195 -v M=308 -v Q=309 '
NR>=N && NR<=M { block[NR-N] = $0 }
NR==Q { print; for (i=0; i<=M-N; i++) print block[i]; next }
{ print }
' $PG_FILE
sed -i '423s/$/ \n/' $PG_FILE

# Change copied function definition
sed -i '195s/^\(\s*\)/\1async /' $PG_FILE
sed -i '195s/insert/ainsert/' $PG_FILE

# Make transformations run asynchronously
sed -i '202d' $PG_FILE
sed -i '205d' $PG_FILE
sed -i '202s/^\(\s*\)/\1await /' $PG_FILE

# Make text embeddings run asynchronously
sed -i '257d' $PG_FILE
sed -i '260d' $PG_FILE
sed -i '257s/^\(\s*\)/\1embeddings = await /' $PG_FILE

# Make KG embeddings run asynchronously
sed -i '272d' $PG_FILE
sed -i '275d' $PG_FILE
sed -i '272s/^\(\s*\)/\1kg_embeddings = await /' $PG_FILE

