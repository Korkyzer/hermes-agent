# Voice Her Prototype

Minimal standalone voice loop for KOR-860:

- browser microphone capture as PCM 16 kHz mono
- FastAPI WebSocket bridge to Gemini Live
- streamed PCM 24 kHz audio playback in the browser
- discreet real-time transcript panel
- basic auth for the page and WebSocket

## Local setup

```bash
cd prototypes/voice-her
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

Required environment:

- `GOOGLE_API_KEY` for API key auth, or `PROJECT_ID` with Google Cloud application default credentials for Vertex auth
- `BASIC_AUTH_USERNAME`
- `BASIC_AUTH_PASSWORD`

Open `http://127.0.0.1:8760`.

## Tailscale deployment

The browser microphone requires a secure context. Use Tailscale Serve to expose the local service over HTTPS on the MagicDNS host:

```bash
sudo cp deploy/voice-her.service /etc/systemd/system/voice-her.service
sudo systemctl daemon-reload
sudo systemctl enable --now voice-her.service
tailscale serve --bg --https=443 http://127.0.0.1:8760
```

Access URL:

```text
https://marquis-vps.tailce955e.ts.net/
```

`voice.hermes.local` can point to the same Tailscale IP if Arthur has local DNS for it, but HTTPS microphone support is safest on the Tailscale MagicDNS name.

Edit `deploy/voice-her.service` if the repo is checked out somewhere other than `/opt/hermes-agent`.
