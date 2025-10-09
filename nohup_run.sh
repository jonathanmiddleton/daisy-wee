#!/usr/bin/env bash

(nohup ./run.sh "$@" </dev/null >nohup.out 2>&1 &); disown