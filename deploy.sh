#!/usr/bin/env bash
# ----------------------------------------------------------
# EC2 Deployment Script — Customer Churn ML
# Run this ONCE after cloning the repo on a fresh Ubuntu EC2.
#   chmod +x deploy.sh && sudo ./deploy.sh
# ----------------------------------------------------------
set -euo pipefail

# -------------------- Config -------------------------
APP_NAME="churn-ml"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"   # repo root
APP_USER="$(logname 2>/dev/null || echo "$SUDO_USER")"
VENV_DIR="$APP_DIR/venv"
DOMAIN="_"                                  # _ = any; replace with your domain
WORKERS=3                                   # gunicorn workers
PORT=5000                                   # internal gunicorn port
# -----------------------------------------------------

echo "====> [1/7] Updating system packages..."
apt-get update -y
apt-get install -y python3 python3-venv python3-pip nginx

echo "====> [2/7] Creating Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv "$VENV_DIR"

echo "====> [3/7] Installing Python dependencies..."
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "====> [4/7] Training model (generates model.pkl)..."
cd "$APP_DIR"
sudo -u "$APP_USER" "$VENV_DIR/bin/python" train.py
if [ ! -f "$APP_DIR/model.pkl" ]; then
    echo "ERROR: model.pkl was not created. Check train.py output above."
    exit 1
fi

echo "====> [5/7] Creating systemd service for gunicorn..."
cat > /etc/systemd/system/${APP_NAME}.service <<EOF
[Unit]
Description=Customer Churn ML (gunicorn)
After=network.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${VENV_DIR}/bin/gunicorn --workers ${WORKERS} --bind 127.0.0.1:${PORT} app:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}
systemctl restart ${APP_NAME}

echo "====> [6/7] Configuring nginx reverse proxy..."
cat > /etc/nginx/sites-available/${APP_NAME} <<EOF
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass         http://127.0.0.1:${PORT};
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }
}
EOF

# Enable site, remove default if it exists
ln -sf /etc/nginx/sites-available/${APP_NAME} /etc/nginx/sites-enabled/${APP_NAME}
rm -f /etc/nginx/sites-enabled/default

nginx -t          # validate config
systemctl enable nginx
systemctl restart nginx

echo "====> [7/7] Deployment complete!"
echo ""
echo "  App running at:  http://<your-ec2-public-ip>"
echo "  Gunicorn logs:   journalctl -u ${APP_NAME} -f"
echo "  Nginx logs:      /var/log/nginx/"
echo ""
echo "  Make sure EC2 Security Group allows inbound HTTP (port 80)."
