# ==================================
# DEVELOPMENT MODE SCRIPT (PowerShell)
# ==================================
# Quick script to run in development mode
# Usage: .\dev.ps1 [command]
# Examples:
#   .\dev.ps1 up        - Start services with hot-reload
#   .\dev.ps1 down      - Stop services
#   .\dev.ps1 logs      - View logs
#   .\dev.ps1 restart   - Restart services

docker-compose -f docker-compose.yml -f docker-compose.dev.yml $args
