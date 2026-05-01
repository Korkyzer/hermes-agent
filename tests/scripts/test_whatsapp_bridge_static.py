from pathlib import Path


BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge" / "bridge.js"


def _src() -> str:
    return BRIDGE.read_text(encoding="utf-8")


def test_whatsapp_bridge_persists_latest_qr_for_headless_pairing():
    src = _src()

    assert "const QR_FILE = getArg('qr-file'" in src
    assert "process.env.WHATSAPP_QR_FILE" in src
    assert "writeFileSync(QR_FILE, qr, 'utf8')" in src


def test_whatsapp_bridge_supports_native_audio_send_media():
    src = _src()

    assert "if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';" in src
    assert "case 'audio':" in src
    assert "audio: buffer" in src
    assert "ptt: ext === 'ogg' || ext === 'opus'" in src
