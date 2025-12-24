#!/bin/bash
# ==================================
# PRODUCTION MODE SCRIPT
# ==================================
# Quick script to run in production mode (for Linode VPS)
# Usage: ./prod.sh [command]
# Examples:
#   ./prod.sh up -d --build   - Deploy/update in production
#   ./prod.sh down            - Stop services
#   ./prod.sh logs -f         - Follow logs
#   ./prod.sh restart         - Restart services

docker compose -f docker-compose.yml -f docker-compose.prod.yml "$@"
