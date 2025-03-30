#!/bin/bash
gunicorn app:app.server -c gunicorn_config.py
