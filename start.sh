#!/bin/bash
gunicorn app.main:server -c gunicorn_config.py
