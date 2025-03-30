#!/bin/bash
gunicorn app:dash_app -c gunicorn_config.py
