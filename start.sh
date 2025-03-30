#!/bin/bash
gunicorn app:app -c gunicorn_config.py
