#!/bin/bash
# bash Script to run UCLCHEMCMC

gnome-terminal -- bash -c "celery worker -A GUI.celery --loglevel=info"
gnome-terminal -- bash -c "python GUI.py"
sudo bash ./run-redis.sh
