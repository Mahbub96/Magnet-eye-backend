# Magnet Eye – Self-Hosting Guide

This guide explains how to run the `magnet-eye-backend` service on your own Linux server, Raspberry Pi, home NAS, or cloud VPS (e.g. Hetzner, DigitalOcean, AWS EC2) using Docker Compose and Caddy for automated HTTPS certificates.

---

## 1. Requirements

- **Operating System:** Linux (Ubuntu 22.04+, Debian 12+, Arch, Fedora, etc.), macOS, or Windows with WSL2.
- **Hardware Resources:**
  - Minimum 1 CPU core.
  - Minimum 1 GB RAM (the Docker container enforces a strict 1024 MB ceiling).
  - 2 GB free disk space.
- **Software:**
  - Docker Engine 24.0+
  - Docker Compose v2 (included in `docker compose`)
- **Network / Domain:**
  - For remote access outside your home Wi-Fi: A public IP and domain name pointing to your server (e.g., `magnet.yourdomain.com` or a free Dynamic DNS hostname from DuckDNS/No-IP), with ports `80` and `443` open.
  - For local LAN use only: You can use your server's local LAN IP (e.g. `192.168.1.100`) directly on port 8001 or using Caddy internal TLS.

---

## 2. Quick Start with Docker Compose

### Step 1: Clone or copy the backend repository

```bash
git clone https://github.com/your-username/magnet-eye.git
cd "Magnet Eye/magnet-eye-backend"
```

### Step 2: Configure Environment Variables

Create a `.env` file in `magnet-eye-backend/`:

```bash
# Set your fully qualified domain name for public automatic HTTPS
DOMAIN=magnet.yourdomain.com
```

*(If testing locally on your LAN without a public domain, you can leave `DOMAIN=localhost` or map port `8001:8001` directly in `docker-compose.yml`)*

### Step 3: Launch Services

```bash
docker compose up -d --build
```

Docker Compose will build the Python image, launch `backend` on port 8001 (isolated internal Docker network), and launch `caddy` which automatically provisions a free Let's Encrypt / ZeroSSL TLS certificate for `$DOMAIN`.

### Step 4: Verify Deployment

Check service health:

```bash
curl -i https://magnet.yourdomain.com/v1/health
```

Expected output:
```json
HTTP/2 200
content-type: application/json

{"status":"ok","api_version":"1.0.0","uptime_s":14.2}
```

View live container logs:
```bash
docker compose logs -f backend
```

---

## 3. Privacy & Resource Guarantees

- **Zero-Storage Privacy:** Magnet Eye stores **nothing** on disk. All sessions and intermediate swipe uploads are maintained in an in-memory session cache that expires 30 minutes after the last activity.
- **No Telemetry / No Logging of Measurements:** Raw magnetic vectors ($B_x, B_y, B_z$) and poses are processed purely in RAM and never written to server log files.
- **Strict 1 GB RAM Ceiling:** The Docker container enforces `limits: memory: 1024M` and `cpus: '1.0'`, ensuring zero interference with your other server workloads.
- **Single Worker Queue:** Inversion solves run sequentially through a bounded queue to prevent CPU saturation. Concurrent excess requests receive standard `HTTP 429 Too Many Requests` with `Retry-After: 2` headers.

---

## 4. Connecting the Phone App

1. Open the **Magnet Eye** app on your Android phone.
2. Navigate to **Settings** (top right gear icon).
3. Under **Backend Service URL**, enter your server endpoint:
   - For public HTTPS: `https://magnet.yourdomain.com`
   - For local LAN: `http://192.168.0.x:8001`
4. Tap **Test Connection**. A green indicator confirms the server is reachable and running compatible API version 1.0.
