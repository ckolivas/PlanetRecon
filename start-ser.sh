#!/bin/sh
schedtool -D -n 19 $$
.venv/bin/python3 planetrecon ser "$@"
