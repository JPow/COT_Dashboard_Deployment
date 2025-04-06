#!/bin/bash
gunicorn app:server -c gunicorn_config.py
