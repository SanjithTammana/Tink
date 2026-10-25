"""Minimal text-only connection check; never print credentials or HTTP bodies."""
import json
from pathlib import Path
import sys
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemini_client import GeminiClient, GeminiError, _https_transport


def main():
    status = []
    def checked(request, timeout, limit):
        try:
            return _https_transport(request, timeout, limit)
        except urllib.error.HTTPError as error:
            status.append(error.code)
            raise
    client = GeminiClient(Path(__file__).resolve().parents[1], transport=checked)
    print('Key configured:', client.configured)
    if not client.configured:
        return 1
    body = json.dumps({'model': client.model, 'store': False,
        'input': 'Connection check only. Return a JSON object with ok true.',
        'generation_config': {'thinking_level': 'low', 'max_output_tokens': 100},
        'response_format': {'type': 'text', 'mime_type': 'application/json', 'schema': {
            'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}}}).encode()
    try:
        result = client._send_body(body, timeout=20.)
    except GeminiError as error:
        print('Gemini connection failed; HTTP status:', status or 'unavailable')
        print(str(error))
        return 1
    print('Gemini connection verified:', result == {'ok': True})
    return 0 if result == {'ok': True} else 1


if __name__ == '__main__':
    raise SystemExit(main())
