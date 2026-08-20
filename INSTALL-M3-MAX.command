#!/bin/zsh

PROJECT_ROOT="${0:A:h}"
BOOTSTRAP="$PROJECT_ROOT/scripts/bootstrap-m3-max.sh"

print "Document Eater setup for M3 Max 36 GB"
print "Project: $PROJECT_ROOT"
print

zsh "$BOOTSTRAP"
STATUS=$?

print
if (( STATUS == 0 )); then
  print "Installation completed successfully."
else
  print -u2 "Installation stopped with exit code $STATUS."
  print -u2 "You can rerun this file; completed model downloads will be reused."
fi
print
read -r "?Press Enter to close this window..."
exit "$STATUS"
