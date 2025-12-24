#!/bin/bash
# ==================================
# DEVELOPMENT MODE SCRIPT
# ==================================
# Quick script to run in development mode
# Usage: ./dev.sh [command]
# Examples:
#   ./dev.sh up        - Start services with hot-reload
#   ./dev.sh down      - Stop services
#   ./dev.sh logs      - View logs
#   ./dev.sh restart   - Restart services

docker compose -f docker-compose.yml -f docker-compose.dev.yml "$@"
